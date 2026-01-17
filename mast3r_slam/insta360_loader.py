"""
Insta360 X5 dataset loader for MASt3R-SLAM.

Streams 512x512 perspective frames directly from dual fisheye video
without intermediate disk storage.

Usage:
    python main.py --dataset insta360:/path/to/video.insv --config config/base.yaml

Or with parameters:
    python main.py --dataset "insta360:/path/to/video.insv?start=10&duration=60&fov=90&fps=10"
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

    def _compute_grid(self, yaw: float) -> Tuple[torch.Tensor, bool]:
        fish_w, fish_h = self.fisheye_size
        out_w, out_h = self.output_size

        normalized_yaw = yaw % 360
        if normalized_yaw > 180:
            normalized_yaw -= 360
        use_front = abs(normalized_yaw) <= 90

        if use_front:
            yaw_rad = np.radians(normalized_yaw)
        else:
            yaw_rad = np.radians(normalized_yaw - 180) if normalized_yaw > 0 else np.radians(normalized_yaw + 180)

        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=torch.float32, device=self.device)

        rays_flat = self.rays_base.reshape(-1, 3)
        rays_rot = (R_yaw @ rays_flat.T).T.reshape(out_h, out_w, 3)

        x, y, z = rays_rot[..., 0], rays_rot[..., 1], rays_rot[..., 2]
        r_xy = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(r_xy, z)

        r = self.f * theta
        phi = torch.atan2(y, x)

        u = r * torch.cos(phi) + fish_w / 2
        v = r * torch.sin(phi) + fish_h / 2

        u_norm = 2.0 * u / (fish_w - 1) - 1.0
        v_norm = 2.0 * v / (fish_h - 1) - 1.0

        valid = (z > 0) & (theta < np.radians(100))
        u_norm = torch.where(valid, u_norm, torch.tensor(2.0, device=self.device))
        v_norm = torch.where(valid, v_norm, torch.tensor(2.0, device=self.device))

        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)
        return grid, use_front

    def extract(self, front_tensor: torch.Tensor, back_tensor: torch.Tensor, yaw: float) -> torch.Tensor:
        grid, use_front = self._compute_grid(yaw)
        source = front_tensor if use_front else back_tensor
        return F.grid_sample(source, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


class Insta360Dataset:
    """
    MASt3R-SLAM compatible dataset for Insta360 X5 dual fisheye video.

    Streams perspectives directly from video without disk I/O.
    """

    def __init__(
        self,
        video_path: str,
        start_time: float = 0,
        duration: float = 60,
        output_size: Tuple[int, int] = (512, 512),
        output_fov: float = 90.0,
        perspectives_per_second: int = 10,
        dtype=np.float32
    ):
        self.dtype = dtype
        self.video_path = video_path
        self.start_time = start_time
        self.duration = duration
        self.output_size = output_size
        self.output_fov = output_fov
        self.perspectives_per_second = perspectives_per_second

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

        # Probe video and set total frames
        self.total_frames = int(duration * perspectives_per_second)
        self._probe_video()

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

        # Compute minimum decode resolution for quality
        out_w = self.output_size[0]
        equirect_w = int(out_w * (360.0 / self.output_fov))
        min_fisheye = int(equirect_w / np.pi * 1.5)
        self.decode_size = ((min_fisheye + 63) // 64) * 64
        self.decode_size = min(self.decode_size, self.native_width)

        print(f"Insta360 X5 Dataset:")
        print(f"  Input: {self.native_width}x{self.native_height} @ {self.input_fps:.1f} fps")
        print(f"  Decode: {self.decode_size}x{self.decode_size}")
        print(f"  Output: {self.output_size[0]}x{self.output_size[1]} @ {self.output_fov}° FOV")
        print(f"  Rate: {self.perspectives_per_second} perspectives/sec")
        print(f"  Total: {self.total_frames} frames over {self.duration}s")

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
            print("Preparing video streams for SLAM...")

            # Extract and scale front stream
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(self.start_time), '-i', self.video_path,
                '-t', str(self.duration), '-map', '0:v:0',
                '-vf', f'scale={self.decode_size}:{self.decode_size}',
                '-c:v', 'hevc_nvenc', '-preset', 'p1', '-rc', 'constqp', '-qp', '18',
                front_path
            ], capture_output=True, check=True)

            # Extract and scale back stream
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(self.start_time), '-i', self.video_path,
                '-t', str(self.duration), '-map', '0:v:1',
                '-vf', f'scale={self.decode_size}:{self.decode_size}',
                '-c:v', 'hevc_nvenc', '-preset', 'p1', '-rc', 'constqp', '-qp', '18',
                back_path
            ], capture_output=True, check=True)

            print("  Streams ready, starting decode...")

            # Create decoders
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

            # Initialize extractor
            extractor = GPUPerspectiveExtractor(
                fisheye_size=(self.decode_size, self.decode_size),
                output_size=self.output_size,
                output_fov=self.output_fov,
                native_size=self.native_width
            )

            frames_per_perspective = self.input_fps / self.perspectives_per_second
            yaw_step = 360.0 / self.perspectives_per_second

            frame_count = 0
            perspective_count = 0

            while not self.stop_event.is_set() and perspective_count < self.total_frames:
                # Decode frame
                front_batch = front_decoder.get_batch_frames(1)
                back_batch = back_decoder.get_batch_frames(1)

                if front_batch is None or back_batch is None:
                    break
                if len(front_batch) == 0 or len(back_batch) == 0:
                    break

                # Check if we should extract from this frame
                if frame_count >= perspective_count * frames_per_perspective:
                    front_frame = front_batch[0]
                    back_frame = back_batch[0]

                    # Convert to tensor
                    def to_tensor(frame):
                        ptr = frame.GetPtrToPlane(0)
                        shape = frame.shape
                        size = int(np.prod(shape))
                        buf = (ctypes.c_uint8 * size).from_address(ptr)
                        arr = np.frombuffer(buf, dtype=np.uint8).reshape(shape).copy()
                        t = torch.from_numpy(arr).to(device).float()
                        return t.permute(2, 0, 1).unsqueeze(0)

                    front_t = to_tensor(front_frame)
                    back_t = to_tensor(back_frame)

                    # Extract forward-facing perspective (yaw=0) for stable SLAM tracking
                    # Spiral rotation causes too much visual change between frames
                    yaw = 0.0  # Always face forward for SLAM
                    perspective = extractor.extract(front_t, back_t, yaw)

                    # Convert to numpy [0, 1] for MASt3R
                    img = perspective.squeeze(0).permute(1, 2, 0)
                    img = img.clamp(0, 255).cpu().numpy() / 255.0
                    img = img.astype(self.dtype)

                    timestamp = self.start_time + perspective_count / self.perspectives_per_second

                    try:
                        self.frame_queue.put((timestamp, img), timeout=1.0)
                        perspective_count += 1
                    except queue.Full:
                        if self.stop_event.is_set():
                            break

                frame_count += 1

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

    Format: insta360:/path/to/video.insv?start=10&duration=60&fov=90&fps=10

    Parameters:
        start: Start time in seconds (default: 0)
        duration: Duration in seconds (default: 60)
        fov: Output FOV in degrees (default: 90)
        fps: Perspectives per second (default: 10)
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
    duration = float(params.get('duration', [60])[0])
    output_fov = float(params.get('fov', [90])[0])
    fps = int(params.get('fps', [10])[0])

    return Insta360Dataset(
        video_path=path,
        start_time=start_time,
        duration=duration,
        output_fov=output_fov,
        perspectives_per_second=fps
    )
