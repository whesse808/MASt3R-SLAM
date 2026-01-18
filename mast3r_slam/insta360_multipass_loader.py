"""
Insta360 X5 multi-pass dataset loader for MASt3R-SLAM.

Multi-pass strategy:
1. Pass 1: Run SLAM on view 0 (forward-facing) for all timestamps
2. Pass 2-N: Run SLAM on views 1-9, using cross-view edges to previous views

This maintains temporal continuity within each view while building
a complete 360° reconstruction.

Usage:
    python main_multipass.py --dataset "insta360:/path/to/video.insv?duration=60&views=10"
"""

import os
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Tuple, List, Optional
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
        native_f = 1100.0
        self.f = native_f * (fisheye_size[0] / native_size)

        self._precompute_base_rays()
        self._grid_cache = {}

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
        # Check cache
        cache_key = round(yaw, 1)
        if cache_key in self._grid_cache:
            grid_front, valid_front, grid_back, valid_back = self._grid_cache[cache_key]
        else:
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

            self._grid_cache[cache_key] = (grid_front, valid_front, grid_back, valid_back)

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
    Returns minimum of front/back differences (camera motion affects both equally).
    """
    h, w = front_curr.shape[2], front_curr.shape[3]
    cy, cx = h // 2, w // 2
    half = window_size // 2

    front_curr_crop = front_curr[:, :, cy-half:cy+half, cx-half:cx+half]
    front_prev_crop = front_prev[:, :, cy-half:cy+half, cx-half:cx+half]
    back_curr_crop = back_curr[:, :, cy-half:cy+half, cx-half:cx+half]
    back_prev_crop = back_prev[:, :, cy-half:cy+half, cx-half:cx+half]

    front_diff = torch.abs(front_curr_crop - front_prev_crop).mean()
    back_diff = torch.abs(back_curr_crop - back_prev_crop).mean()

    return min(front_diff.item(), back_diff.item())


class Insta360MultiPassDataset:
    """
    Multi-pass Insta360 dataset for sequential view processing.

    Processes one view at a time through all timestamps, enabling
    temporal continuity for SLAM tracking.

    Frame ordering: All timestamps for view 0, then all for view 1, etc.
    [t0_v0, t1_v0, t2_v0, ..., t0_v1, t1_v1, t2_v1, ...]
    """

    def __init__(
        self,
        video_path: str,
        start_time: float = 0,
        duration: float = 60,
        output_size: Tuple[int, int] = (512, 512),
        output_fov: float = 90.0,
        fps: int = 10,
        num_views: int = 10,
        motion_threshold: float = 0.0,
        motion_window: int = 64,
        dtype=np.float32
    ):
        self.dtype = dtype
        self.video_path = video_path
        self.start_time = start_time
        self.duration = duration
        self.output_size = output_size
        self.output_fov = output_fov
        self.fps = fps
        self.num_views = num_views
        self.motion_threshold = motion_threshold
        self.motion_window = motion_window

        # MASt3R-SLAM interface
        self.rgb_files = []
        self.timestamps = []
        self.img_size = 512
        self.camera_intrinsics = None
        self.use_calibration = False
        self.save_results = True
        self.dataset_path = Path(video_path).parent

        # Frame metadata: (timestamp, view_idx, yaw)
        self.frame_metadata: List[Tuple[float, int, float]] = []

        # Streaming
        self.frame_queue = queue.Queue(maxsize=30)
        self.stop_event = threading.Event()
        self.decoder_thread = None
        self.frames_yielded = 0

        # Probe and compute frame count
        self._probe_video()

        # Compute view yaws (evenly spaced around 360°)
        self.view_yaws = [i * (360.0 / num_views) for i in range(num_views)]

        # Pre-compute frame ordering
        self._compute_frame_order()

    def _probe_video(self):
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'stream=index,codec_type,width,height,r_frame_rate',
            '-of', 'csv=p=0', self.video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)

        video_streams = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 5 and parts[1] == 'video':
                video_streams.append({
                    'width': int(parts[2]),
                    'height': int(parts[3]),
                    'fps': parts[4]
                })

        if len(video_streams) < 2:
            raise ValueError(f"Expected 2 video streams in Insta360 file, found {len(video_streams)}")

        self.native_width = video_streams[0]['width']
        self.native_height = video_streams[0]['height']
        fps_num, fps_den = map(int, video_streams[0]['fps'].split('/'))
        self.input_fps = fps_num / fps_den

        # Compute decode resolution
        out_w = self.output_size[0]
        equirect_w = int(out_w * (360.0 / self.output_fov))
        min_fisheye = int(equirect_w / np.pi * 1.5)
        self.decode_size = ((min_fisheye + 63) // 64) * 64
        self.decode_size = min(self.decode_size, self.native_width)

        # Max timestamps based on duration and fps
        self.max_timestamps = int(self.duration * self.fps)

    def _compute_frame_order(self):
        """Pre-compute frame order: all timestamps for each view in sequence."""
        self.frame_metadata = []

        # VIEW-FIRST ordering: complete each view before moving to next
        for view_idx in range(self.num_views):
            yaw = self.view_yaws[view_idx]
            for t_idx in range(self.max_timestamps):
                timestamp = self.start_time + t_idx / self.fps
                self.frame_metadata.append((timestamp, view_idx, yaw))

        self.total_frames = len(self.frame_metadata)

        print(f"Insta360 X5 Multi-Pass Dataset:")
        print(f"  Input: {self.native_width}x{self.native_height} @ {self.input_fps:.1f} fps")
        print(f"  Decode: {self.decode_size}x{self.decode_size}")
        print(f"  Output: {self.output_size[0]}x{self.output_size[1]} @ {self.output_fov}° FOV")
        print(f"  Views: {self.num_views} (36° apart)")
        print(f"  Timestamps: {self.max_timestamps} @ {self.fps} fps")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Ordering: VIEW-FIRST (all timestamps for view 0, then view 1, ...)")
        if self.motion_threshold > 0:
            print(f"  Motion detection: {self.motion_window}x{self.motion_window} window, threshold={self.motion_threshold}")

    def _start_streaming(self):
        if self.decoder_thread is not None:
            return
        self.decoder_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.decoder_thread.start()

    def _decode_loop(self):
        import tempfile
        import shutil

        device = torch.device('cuda')
        temp_dir = tempfile.mkdtemp()
        front_path = os.path.join(temp_dir, 'front.mp4')
        back_path = os.path.join(temp_dir, 'back.mp4')

        try:
            print("Preparing video streams...")

            # Extract and scale streams
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(self.start_time), '-i', self.video_path,
                '-t', str(self.duration), '-map', '0:v:0',
                '-vf', f'scale={self.decode_size}:{self.decode_size}',
                '-c:v', 'hevc_nvenc', '-preset', 'p1', '-rc', 'constqp', '-qp', '18',
                front_path
            ], capture_output=True, check=True)

            subprocess.run([
                'ffmpeg', '-y', '-ss', str(self.start_time), '-i', self.video_path,
                '-t', str(self.duration), '-map', '0:v:1',
                '-vf', f'scale={self.decode_size}:{self.decode_size}',
                '-c:v', 'hevc_nvenc', '-preset', 'p1', '-rc', 'constqp', '-qp', '18',
                back_path
            ], capture_output=True, check=True)

            print("  Streams ready, decoding all frames into memory...")

            if not HAS_PYNVVIDEOCODEC:
                raise RuntimeError("PyNvVideoCodec required")

            front_decoder = nvc.CreateSimpleDecoder(
                encSource=front_path, gpuid=0, useDeviceMemory=False,
                outputColorType=nvc.OutputColorType.RGB, decoderCacheSize=1
            )
            back_decoder = nvc.CreateSimpleDecoder(
                encSource=back_path, gpuid=0, useDeviceMemory=False,
                outputColorType=nvc.OutputColorType.RGB, decoderCacheSize=1
            )

            extractor = GPUPerspectiveExtractor(
                fisheye_size=(self.decode_size, self.decode_size),
                output_size=self.output_size,
                output_fov=self.output_fov,
                native_size=self.native_width
            )

            def to_tensor(frame):
                ptr = frame.GetPtrToPlane(0)
                shape = frame.shape
                size = int(np.prod(shape))
                buf = (ctypes.c_uint8 * size).from_address(ptr)
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(shape).copy()
                t = torch.from_numpy(arr).to(device).float()
                return t.permute(2, 0, 1).unsqueeze(0)

            # Decode all frames at target fps, storing fisheye pairs
            frames_per_sample = self.input_fps / self.fps
            fisheye_frames: List[Tuple[torch.Tensor, torch.Tensor]] = []

            frame_count = 0
            sample_count = 0
            prev_front_t = None
            prev_back_t = None
            skipped_count = 0

            # Track which timestamps have motion (for filtering)
            valid_timestamps: List[int] = []

            while sample_count < self.max_timestamps:
                front_batch = front_decoder.get_batch_frames(1)
                back_batch = back_decoder.get_batch_frames(1)

                if front_batch is None or back_batch is None:
                    break
                if len(front_batch) == 0 or len(back_batch) == 0:
                    break

                target_frame = int(sample_count * frames_per_sample)
                if frame_count >= target_frame:
                    front_t = to_tensor(front_batch[0])
                    back_t = to_tensor(back_batch[0])

                    # Motion detection
                    has_motion = True
                    if self.motion_threshold > 0 and prev_front_t is not None:
                        motion_score = compute_motion_score(
                            front_t, back_t, prev_front_t, prev_back_t,
                            window_size=self.motion_window
                        )
                        has_motion = motion_score >= self.motion_threshold

                    if has_motion:
                        # Store fisheye pair for this timestamp
                        fisheye_frames.append((front_t.cpu(), back_t.cpu()))
                        valid_timestamps.append(sample_count)
                        prev_front_t = front_t
                        prev_back_t = back_t
                    else:
                        skipped_count += 1

                    sample_count += 1

                frame_count += 1

            n_valid = len(fisheye_frames)
            print(f"  Decoded {n_valid} timestamps (skipped {skipped_count} static)")

            # Now generate perspectives in VIEW-FIRST order
            print(f"  Generating {n_valid * self.num_views} perspectives...")

            perspective_count = 0
            for view_idx in range(self.num_views):
                if self.stop_event.is_set():
                    break

                yaw = self.view_yaws[view_idx]
                print(f"    View {view_idx}: yaw={yaw:.0f}°")

                for t_idx, (front_t, back_t) in enumerate(fisheye_frames):
                    if self.stop_event.is_set():
                        break

                    # Move tensors to GPU for extraction
                    front_gpu = front_t.to(device)
                    back_gpu = back_t.to(device)

                    # Extract perspective
                    perspective = extractor.extract(front_gpu, back_gpu, yaw)

                    # Convert to numpy
                    img = perspective.squeeze(0).permute(1, 2, 0)
                    img = img.clamp(0, 255).cpu().numpy() / 255.0
                    img = img.astype(self.dtype)

                    # Timestamp from original valid timestamp index
                    orig_t_idx = valid_timestamps[t_idx]
                    timestamp = self.start_time + orig_t_idx / self.fps

                    # Metadata: (timestamp, view_idx, yaw)
                    metadata = {
                        'timestamp': timestamp,
                        'view_idx': view_idx,
                        'yaw': yaw,
                        't_idx': t_idx,
                        'frame_idx': perspective_count
                    }

                    try:
                        self.frame_queue.put((timestamp, img, metadata), timeout=1.0)
                        perspective_count += 1
                    except queue.Full:
                        if self.stop_event.is_set():
                            break

            print(f"  Decoder finished: {perspective_count} perspectives")

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

        timestamp, img, metadata = item
        self.timestamps.append(timestamp)
        self.frames_yielded += 1
        return timestamp, img

    def get_frame_metadata(self, idx):
        """Get metadata for frame at index."""
        if idx < len(self.frame_metadata):
            return self.frame_metadata[idx]
        return None

    def get_timestamp(self, idx):
        if idx < len(self.timestamps):
            return self.timestamps[idx]
        return idx / self.fps

    def get_img_shape(self):
        return (384, 512), self.output_size

    def subsample(self, subsample):
        pass

    def has_calib(self):
        return False

    def stop(self):
        self.stop_event.set()
        if self.decoder_thread:
            self.decoder_thread.join(timeout=5.0)

    def get_dataset_info(self):
        """Return info for multi-view backend."""
        return {
            'num_views': self.num_views,
            'views_per_timestamp': self.num_views,
            'n_timestamps': self.max_timestamps,
            'view_yaws': self.view_yaws,
            'ordering': 'view_first'
        }


def parse_insta360_multipass_dataset(dataset_path: str) -> Insta360MultiPassDataset:
    """
    Parse dataset path with parameters.

    Format: insta360mp:/path/to/video.insv?start=10&duration=60&views=10&fps=5&motion_thresh=5
    """
    # Remove prefix
    if dataset_path.startswith('insta360mp:'):
        dataset_path = dataset_path[11:]

    # Parse URL-style parameters
    if '?' in dataset_path:
        path, query = dataset_path.split('?', 1)
        params = parse_qs(query)
    else:
        path = dataset_path
        params = {}

    start_time = float(params.get('start', [0])[0])
    duration = float(params.get('duration', [60])[0])
    output_fov = float(params.get('fov', [90])[0])
    fps = int(params.get('fps', [10])[0])
    num_views = int(params.get('views', [10])[0])
    motion_threshold = float(params.get('motion_thresh', [0])[0])
    motion_window = int(params.get('motion_window', [64])[0])

    return Insta360MultiPassDataset(
        video_path=path,
        start_time=start_time,
        duration=duration,
        output_fov=output_fov,
        fps=fps,
        num_views=num_views,
        motion_threshold=motion_threshold,
        motion_window=motion_window
    )
