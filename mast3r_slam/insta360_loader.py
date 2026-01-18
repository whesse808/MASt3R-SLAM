"""
Insta360 X5 dataset loader for MASt3R-SLAM.

Streams 512x512 perspective frames directly from dual fisheye video
without intermediate disk storage.

Features:
- Motion detection: skips frames with no camera movement
- Uses 64x64 center crop from both fisheyes (ignores moving objects at edges)

Usage:
    python main.py --dataset insta360:/path/to/video.insv --config config/base.yaml

Or with parameters:
    python main.py --dataset "insta360:/path/to/video.insv?start=10&duration=60&fov=90&fps=10&motion_thresh=5"
"""

import os
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
import torch.nn.functional as F
import subprocess
import ctypes

try:
    import PyNvVideoCodec as nvc
    HAS_PYNVVIDEOCODEC = True
except ImportError:
    HAS_PYNVVIDEOCODEC = False


class GPUPerspectiveExtractor:
    """GPU-resident fisheye to perspective converter."""

    def __init__(
        self,
        fisheye_size: Tuple[int, int],
        output_size: Tuple[int, int] = (512, 512),
        output_fov: float = 90.0,
        device: str = 'cuda',
        native_size: int = 3840
    ):
        self.device = torch.device(device)
        self.fisheye_size = fisheye_size
        self.output_size = output_size
        self.output_fov = output_fov

        # Equidistant fisheye: r = f * theta
        # At edge: r = 1920, theta = 100° = 1.745 rad => f = 1100 for 3840x3840
        native_f = 1100.0
        self.f = native_f * (fisheye_size[0] / native_size)

        self._precompute_base_rays()

    def _precompute_base_rays(self):
        out_w, out_h = self.output_size
        fov_rad = np.radians(self.output_fov)
        f_out = out_w / (2 * np.tan(fov_rad / 2))

        y_out, x_out = torch.meshgrid(
            torch.arange(out_h, dtype=torch.float32, device=self.device),
            torch.arange(out_w, dtype=torch.float32, device=self.device),
            indexing='ij'
        )

        cx, cy = out_w / 2, out_h / 2
        self.rays_base = torch.stack([
            (x_out - cx) / f_out,
            (y_out - cy) / f_out,
            torch.ones_like(x_out)
        ], dim=-1)
        self.rays_base = self.rays_base / torch.norm(self.rays_base, dim=-1, keepdim=True)

    def _compute_grid_for_fisheye(self, yaw_rad: float, for_front: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sampling grid for a single fisheye.

        Returns: (grid, validity_mask) where validity indicates pixels that can be sampled
        """
        fish_w, fish_h = self.fisheye_size
        out_w, out_h = self.output_size

        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=torch.float32, device=self.device)

        rays_flat = self.rays_base.reshape(-1, 3)
        rays_rot = (R_yaw @ rays_flat.T).T.reshape(out_h, out_w, 3)

        x, y, z = rays_rot[..., 0], rays_rot[..., 1], rays_rot[..., 2]

        if for_front:
            # Front fisheye looks along +Z
            r_xy = torch.sqrt(x**2 + y**2)
            theta = torch.atan2(r_xy, z)
            phi = torch.atan2(y, x)
            valid = (z > 0) & (theta < np.radians(100))
        else:
            # Back fisheye looks along -Z
            r_xy = torch.sqrt(x**2 + y**2)
            theta = torch.atan2(r_xy, -z)
            phi = torch.atan2(y, -x)  # flip x for mirror
            valid = (z < 0) & (theta < np.radians(100))

        r = self.f * theta

        u = r * torch.cos(phi) + fish_w / 2
        v = r * torch.sin(phi) + fish_h / 2

        u_norm = 2.0 * u / (fish_w - 1) - 1.0
        v_norm = 2.0 * v / (fish_h - 1) - 1.0

        # Set invalid pixels to out-of-bounds
        u_norm = torch.where(valid, u_norm, torch.tensor(2.0, device=self.device))
        v_norm = torch.where(valid, v_norm, torch.tensor(2.0, device=self.device))

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
        return grid, valid

    def extract(self, front_tensor: torch.Tensor, back_tensor: torch.Tensor, yaw: float) -> torch.Tensor:
        """
        Extract perspective view, blending both fisheyes at the seams.

        At yaw=0° and yaw=180° (the seams between fisheyes), we sample from both
        and blend based on which has valid data.
        """
        # Camera mounted perpendicular to driving direction
        # 270° offset aligns yaw=0 with driving direction
        adjusted_yaw = (yaw + 270.0) % 360

        # Normalize to [-180, 180]
        normalized_yaw = adjusted_yaw
        if normalized_yaw > 180:
            normalized_yaw -= 360

        # Convert to radians for perspective projection
        yaw_rad = np.radians(normalized_yaw)

        # Compute grids for both fisheyes
        grid_front, valid_front = self._compute_grid_for_fisheye(yaw_rad, for_front=True)
        grid_back, valid_back = self._compute_grid_for_fisheye(yaw_rad, for_front=False)

        # Sample from both fisheyes
        sample_front = F.grid_sample(front_tensor, grid_front, mode='bilinear',
                                      padding_mode='zeros', align_corners=True)
        sample_back = F.grid_sample(back_tensor, grid_back, mode='bilinear',
                                     padding_mode='zeros', align_corners=True)

        # Create blend mask: where only one is valid, use that one
        # Where both are valid, blend based on distance from seam
        valid_front_mask = valid_front.unsqueeze(0).unsqueeze(0).float()
        valid_back_mask = valid_back.unsqueeze(0).unsqueeze(0).float()

        # Blend: prioritize based on validity
        both_valid = valid_front_mask * valid_back_mask
        only_front = valid_front_mask * (1 - valid_back_mask)
        only_back = (1 - valid_front_mask) * valid_back_mask

        # Where both valid, blend 50/50 (could use distance-based weights)
        result = (sample_front * (only_front + both_valid * 0.5) +
                  sample_back * (only_back + both_valid * 0.5))

        return result


def compute_motion_score(front_curr: torch.Tensor, back_curr: torch.Tensor,
                         front_prev: torch.Tensor, back_prev: torch.Tensor,
                         window_size: int = 64) -> float:
    """
    Compute camera motion score using center crops of both fisheyes.

    Uses a small center window to focus on static scene elements and
    ignore moving objects that tend to be at the edges (people, cars).

    Returns minimum of front/back differences (camera motion affects both equally).
    """
    h, w = front_curr.shape[2], front_curr.shape[3]
    cy, cx = h // 2, w // 2
    half = window_size // 2

    # Extract center crops
    front_curr_crop = front_curr[:, :, cy-half:cy+half, cx-half:cx+half]
    front_prev_crop = front_prev[:, :, cy-half:cy+half, cx-half:cx+half]
    back_curr_crop = back_curr[:, :, cy-half:cy+half, cx-half:cx+half]
    back_prev_crop = back_prev[:, :, cy-half:cy+half, cx-half:cx+half]

    # Compute mean absolute difference
    front_diff = torch.abs(front_curr_crop - front_prev_crop).mean()
    back_diff = torch.abs(back_curr_crop - back_prev_crop).mean()

    # Use minimum (camera motion affects both, moving objects affect one)
    return min(front_diff.item(), back_diff.item())


class Insta360Dataset:
    """
    MASt3R-SLAM compatible dataset for Insta360 X5 dual fisheye video.

    Streams perspectives directly from video without disk I/O.
    """

    def __init__(
        self,
        video_path: str,
        start_time: float = 0,
        duration: float = 0,  # 0 = scan entire video
        output_size: Tuple[int, int] = (512, 512),
        output_fov: float = 90.0,
        perspectives_per_second: int = 10,
        motion_threshold: float = 0.0,  # 0 = disabled, 5.0 = typical threshold
        motion_window: int = 64,
        max_motion_frames: int = 30,  # Stop after collecting this many motion frames
        dtype=np.float32
    ):
        self.dtype = dtype
        self.video_path = video_path
        self.start_time = start_time
        self.duration = duration  # 0 = unlimited (scan until max_motion_frames)
        self.output_size = output_size
        self.output_fov = output_fov
        self.perspectives_per_second = perspectives_per_second
        self.motion_threshold = motion_threshold
        self.motion_window = motion_window
        self.max_motion_frames = max_motion_frames

        # MASt3R-SLAM interface
        self.rgb_files = []
        self.timestamps = []
        self.img_size = 512
        self.camera_intrinsics = None
        self.use_calibration = False
        self.save_results = True
        self.dataset_path = Path(video_path).parent

        # Streaming
        self.frame_queue = queue.Queue(maxsize=30)
        self.stop_event = threading.Event()
        self.decoder_thread = None

        # Total frames is the target motion frames
        self.total_frames = max_motion_frames
        self._probe_video()

    def _probe_video(self):
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'stream=index,codec_type,width,height,r_frame_rate',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0', self.video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)

        video_streams = []
        self.video_duration = 0
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 5 and parts[1] == 'video':
                video_streams.append({
                    'width': int(parts[2]),
                    'height': int(parts[3]),
                    'fps': parts[4]
                })
            elif len(parts) == 1:
                try:
                    self.video_duration = float(parts[0])
                except ValueError:
                    pass

        if len(video_streams) < 2:
            raise ValueError(f"Expected 2 video streams in Insta360 file, found {len(video_streams)}")

        self.native_width = video_streams[0]['width']
        self.native_height = video_streams[0]['height']
        fps_num, fps_den = map(int, video_streams[0]['fps'].split('/'))
        self.input_fps = fps_num / fps_den

        # If duration not specified, use entire video (minus start time)
        if self.duration <= 0:
            self.duration = max(1, self.video_duration - self.start_time)

        # Compute minimum decode resolution for quality
        out_w = self.output_size[0]
        equirect_w = int(out_w * (360.0 / self.output_fov))
        min_fisheye = int(equirect_w / np.pi * 1.5)
        self.decode_size = ((min_fisheye + 63) // 64) * 64
        self.decode_size = min(self.decode_size, self.native_width)

        print(f"Insta360 X5 Dataset:")
        print(f"  Input: {self.native_width}x{self.native_height} @ {self.input_fps:.1f} fps")
        print(f"  Video duration: {self.video_duration:.1f}s, scanning from {self.start_time}s")
        print(f"  Decode: {self.decode_size}x{self.decode_size}")
        print(f"  Output: {self.output_size[0]}x{self.output_size[1]} @ {self.output_fov}° FOV")
        print(f"  Target: {self.max_motion_frames} motion frames @ {self.perspectives_per_second} fps sampling")
        if self.motion_threshold > 0:
            print(f"  Motion detection: {self.motion_window}x{self.motion_window} window, threshold={self.motion_threshold}")

    def _start_streaming(self):
        if self.decoder_thread is not None:
            return
        self.decoder_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.decoder_thread.start()

    def _decode_loop(self):
        """
        Chunked extraction: process video in 30-second segments.
        Exits early once max_motion_frames are collected.
        """
        import tempfile
        import shutil

        device = torch.device('cuda')
        temp_dir = tempfile.mkdtemp()
        front_path = os.path.join(temp_dir, 'front.mp4')
        back_path = os.path.join(temp_dir, 'back.mp4')

        # Initialize extractor once
        extractor = GPUPerspectiveExtractor(
            fisheye_size=(self.decode_size, self.decode_size),
            output_size=self.output_size,
            output_fov=self.output_fov,
            native_size=self.native_width
        )

        frames_per_perspective = self.input_fps / self.perspectives_per_second
        chunk_duration = 30.0  # Process 30 seconds at a time

        total_frame_count = 0
        sample_count = 0
        motion_frame_count = 0
        skipped_count = 0
        prev_front_t = None
        prev_back_t = None

        # Helper to convert decoded frame to tensor
        def to_tensor(frame):
            ptr = frame.GetPtrToPlane(0)
            shape = frame.shape
            size = int(np.prod(shape))
            buf = (ctypes.c_uint8 * size).from_address(ptr)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(shape).copy()
            t = torch.from_numpy(arr).to(device).float()
            return t.permute(2, 0, 1).unsqueeze(0)

        try:
            current_time = self.start_time
            max_time = self.start_time + self.duration

            print(f"Chunked extraction: scanning video for {self.max_motion_frames} motion frames...")

            # Process video in chunks
            while not self.stop_event.is_set() and motion_frame_count < self.max_motion_frames and current_time < max_time:
                chunk_end = min(current_time + chunk_duration, max_time)
                actual_chunk = chunk_end - current_time

                print(f"  Extracting {current_time:.0f}s-{chunk_end:.0f}s ({motion_frame_count}/{self.max_motion_frames} frames)...")

                # Extract this chunk only
                subprocess.run([
                    'ffmpeg', '-y', '-ss', str(current_time), '-i', self.video_path,
                    '-t', str(actual_chunk), '-map', '0:v:0',
                    '-vf', f'scale={self.decode_size}:{self.decode_size}',
                    '-c:v', 'hevc_nvenc', '-preset', 'p1', '-rc', 'constqp', '-qp', '18',
                    front_path
                ], capture_output=True, check=True)

                subprocess.run([
                    'ffmpeg', '-y', '-ss', str(current_time), '-i', self.video_path,
                    '-t', str(actual_chunk), '-map', '0:v:1',
                    '-vf', f'scale={self.decode_size}:{self.decode_size}',
                    '-c:v', 'hevc_nvenc', '-preset', 'p1', '-rc', 'constqp', '-qp', '18',
                    back_path
                ], capture_output=True, check=True)

                # Decode this chunk
                if not HAS_PYNVVIDEOCODEC:
                    raise RuntimeError("PyNvVideoCodec required for GPU decoding")

                front_decoder = nvc.CreateSimpleDecoder(
                    encSource=front_path, gpuid=0, useDeviceMemory=False,
                    outputColorType=nvc.OutputColorType.RGB, decoderCacheSize=1
                )
                back_decoder = nvc.CreateSimpleDecoder(
                    encSource=back_path, gpuid=0, useDeviceMemory=False,
                    outputColorType=nvc.OutputColorType.RGB, decoderCacheSize=1
                )

                chunk_frame = 0
                while not self.stop_event.is_set() and motion_frame_count < self.max_motion_frames:
                    front_batch = front_decoder.get_batch_frames(1)
                    back_batch = back_decoder.get_batch_frames(1)

                    if front_batch is None or back_batch is None or len(front_batch) == 0 or len(back_batch) == 0:
                        break

                    # Check if we should sample this frame
                    target_frame = int(sample_count * frames_per_perspective)
                    if total_frame_count >= target_frame:
                        front_t = to_tensor(front_batch[0])
                        back_t = to_tensor(back_batch[0])

                        has_motion = True
                        if self.motion_threshold > 0 and prev_front_t is not None:
                            motion_score = compute_motion_score(
                                front_t, back_t, prev_front_t, prev_back_t,
                                window_size=self.motion_window
                            )
                            has_motion = motion_score >= self.motion_threshold

                        sample_count += 1

                        if not has_motion:
                            skipped_count += 1
                            total_frame_count += 1
                            chunk_frame += 1
                            continue

                        prev_front_t = front_t.clone()
                        prev_back_t = back_t.clone()

                        yaw = 0.0
                        perspective = extractor.extract(front_t, back_t, yaw)

                        img = perspective.squeeze(0).permute(1, 2, 0)
                        img = img.clamp(0, 255).cpu().numpy() / 255.0
                        img = img.astype(self.dtype)

                        timestamp = current_time + chunk_frame / self.input_fps

                        try:
                            self.frame_queue.put((timestamp, img), timeout=1.0)
                            motion_frame_count += 1
                        except queue.Full:
                            if self.stop_event.is_set():
                                break

                    total_frame_count += 1
                    chunk_frame += 1

                # Clean up decoders before next chunk
                del front_decoder, back_decoder

                current_time = chunk_end

            video_time_scanned = current_time - self.start_time
            if self.motion_threshold > 0:
                print(f"  Done: {motion_frame_count} motion frames (skipped {skipped_count}) from {video_time_scanned:.1f}s of video")
            else:
                print(f"  Done: {motion_frame_count} perspectives from {video_time_scanned:.1f}s")

        except Exception as e:
            print(f"Decoder error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.frame_queue.put(None)

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        if self.decoder_thread is None:
            self._start_streaming()

        item = self.frame_queue.get()
        if item is None:
            raise StopIteration("End of stream")

        timestamp, img = item
        self.timestamps.append(timestamp)
        return timestamp, img

    def get_timestamp(self, idx):
        if idx < len(self.timestamps):
            return self.timestamps[idx]
        return idx / self.perspectives_per_second

    def read_img(self, idx):
        _, img = self.__getitem__(idx)
        return (img * 255).astype(np.uint8)

    def get_image(self, idx):
        _, img = self.__getitem__(idx)
        return img

    def get_img_shape(self):
        # MASt3R's resize_img forces 3:4 aspect ratio for square images
        # For 512x512 input: resized to 512x512, then cropped to 512x384 (h, w)
        return (384, 512), self.output_size

    def subsample(self, subsample):
        pass

    def has_calib(self):
        return False

    def stop(self):
        self.stop_event.set()
        if self.decoder_thread:
            self.decoder_thread.join(timeout=5.0)


def parse_insta360_dataset(dataset_path: str) -> Insta360Dataset:
    """
    Parse dataset path with optional parameters.

    Format: insta360:/path/to/video.insv?start=10&frames=30&fov=90&fps=3&motion_thresh=3

    Parameters:
        start: Start time in seconds (default: 0)
        frames: Number of motion-containing frames to collect (default: 30)
        fov: Output FOV in degrees (default: 90)
        fps: Sampling rate in perspectives per second (default: 3)
        motion_thresh: Motion threshold (0=disabled, 3.0=typical) (default: 3)
        motion_window: Motion detection window size (default: 64)
    """
    # Remove 'insta360:' prefix
    if dataset_path.startswith('insta360:'):
        dataset_path = dataset_path[9:]

    # Parse URL-style parameters
    if '?' in dataset_path:
        path, query = dataset_path.split('?', 1)
        params = parse_qs(query)
    else:
        path = dataset_path
        params = {}

    # Extract parameters with defaults
    start_time = float(params.get('start', [0])[0])
    max_motion_frames = int(params.get('frames', [30])[0])
    output_fov = float(params.get('fov', [90])[0])
    fps = int(params.get('fps', [3])[0])
    motion_threshold = float(params.get('motion_thresh', [3])[0])
    motion_window = int(params.get('motion_window', [64])[0])

    return Insta360Dataset(
        video_path=path,
        start_time=start_time,
        duration=0,  # Scan entire video until max_motion_frames collected
        output_fov=output_fov,
        perspectives_per_second=fps,
        motion_threshold=motion_threshold,
        motion_window=motion_window,
        max_motion_frames=max_motion_frames
    )
