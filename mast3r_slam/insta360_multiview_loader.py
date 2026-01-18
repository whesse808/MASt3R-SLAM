"""
Insta360 X5 Multi-View dataset loader for MASt3R-SLAM.

Outputs multiple perspectives per timestamp with VIEW-FIRST ordering
so consecutive frames look in the same direction (critical for tracking).

Features:
- Motion detection: skips frames with no camera movement
- Uses 64x64 center crop from both fisheyes to detect motion (ignores moving objects at edges)
- VIEW-FIRST ordering for reliable tracking

Frame ordering: [v0_t0, v0_t1, v0_t2, ..., v1_t0, v1_t1, ...]

Usage:
    python main_multiview.py --dataset "insta360mv:/path/to/video.insv?views=10&fps=5&motion_thresh=5"
"""

import os
import queue
import threading
import subprocess
import ctypes
from pathlib import Path
from typing import Tuple, List, Optional
from urllib.parse import parse_qs

import numpy as np
import torch
import torch.nn.functional as F

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


def compute_motion_score(front_curr: torch.Tensor, back_curr: torch.Tensor,
                         front_prev: torch.Tensor, back_prev: torch.Tensor,
                         window_size: int = 64) -> float:
    """
    Compute camera motion score using center crops of both fisheyes.

    Uses a small center window to focus on static scene elements and
    ignore moving objects that tend to be at the edges (people, cars).

    Args:
        front_curr, back_curr: Current frame fisheyes [1, 3, H, W]
        front_prev, back_prev: Previous frame fisheyes [1, 3, H, W]
        window_size: Size of center crop for comparison

    Returns:
        Motion score (mean absolute difference in center region)
    """
    h, w = front_curr.shape[2], front_curr.shape[3]
    cy, cx = h // 2, w // 2
    half = window_size // 2

    # Extract center crops
    front_curr_crop = front_curr[:, :, cy-half:cy+half, cx-half:cx+half]
    front_prev_crop = front_prev[:, :, cy-half:cy+half, cx-half:cx+half]
    back_curr_crop = back_curr[:, :, cy-half:cy+half, cx-half:cx+half]
    back_prev_crop = back_prev[:, :, cy-half:cy+half, cx-half:cx+half]

    # Compute mean absolute difference for both fisheyes
    front_diff = torch.abs(front_curr_crop - front_prev_crop).mean()
    back_diff = torch.abs(back_curr_crop - back_prev_crop).mean()

    # Use minimum of both (camera motion affects both equally)
    # Moving objects would only affect one fisheye significantly
    motion_score = min(front_diff.item(), back_diff.item())

    return motion_score


class Insta360MultiViewDataset:
    """
    MASt3R-SLAM dataset with VIEW-FIRST frame ordering and motion detection.

    Features:
    - Skips frames with no camera movement (reduces redundant processing)
    - Uses 64x64 center crop motion detection (ignores moving objects)
    - VIEW-FIRST ordering for reliable frame-to-frame tracking

    Frame ordering: [v0_t0, v0_t1, v0_t2, ..., v1_t0, v1_t1, v1_t2, ...]
    """

    def __init__(
        self,
        video_path: str,
        start_time: float = 0,
        duration: float = 60,
        output_size: Tuple[int, int] = (512, 512),
        output_fov: float = 90.0,
        views_per_timestamp: int = 10,
        timestamps_per_second: float = 5.0,
        motion_threshold: float = 5.0,  # Skip frames below this motion score
        motion_window: int = 64,  # Size of center crop for motion detection
        dtype=np.float32
    ):
        self.dtype = dtype
        self.video_path = video_path
        self.start_time = start_time
        self.duration = duration
        self.output_size = output_size
        self.output_fov = output_fov
        self.views_per_timestamp = views_per_timestamp
        self.timestamps_per_second = timestamps_per_second
        self.motion_threshold = motion_threshold
        self.motion_window = motion_window

        # Compute yaw angles for each view
        self.yaw_angles = [i * (360.0 / views_per_timestamp) for i in range(views_per_timestamp)]

        # MASt3R-SLAM interface
        self.rgb_files = []
        self.timestamps = []
        self.img_size = 512
        self.camera_intrinsics = None
        self.use_calibration = False
        self.save_results = True
        self.dataset_path = Path(video_path).parent

        # Will be set after motion filtering
        self.n_timestamps = 0
        self.total_frames = 0

        # Pre-extracted frames storage
        self.frames_buffer = None  # Will be list of (views, timestamps with motion)
        self.motion_timestamps = []  # Original timestamp indices that had motion
        self.extraction_done = False

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

        # Compute decode resolution
        out_w = self.output_size[0]
        equirect_w = int(out_w * (360.0 / self.output_fov))
        min_fisheye = int(equirect_w / np.pi * 1.5)
        self.decode_size = ((min_fisheye + 63) // 64) * 64
        self.decode_size = min(self.decode_size, self.native_width)

        # Max timestamps before filtering
        self.max_timestamps = int(self.duration * self.timestamps_per_second)

        print(f"Insta360 X5 Multi-View Dataset (with motion detection):")
        print(f"  Input: {self.native_width}x{self.native_height} @ {self.input_fps:.1f} fps")
        print(f"  Decode: {self.decode_size}x{self.decode_size}")
        print(f"  Output: {self.output_size[0]}x{self.output_size[1]} @ {self.output_fov}° FOV")
        print(f"  Views: {self.views_per_timestamp} (yaw: {self.yaw_angles})")
        print(f"  Max timestamps: {self.max_timestamps} over {self.duration}s")
        print(f"  Motion detection: {self.motion_window}x{self.motion_window} window, threshold={self.motion_threshold}")

    def get_frame_info(self, idx: int) -> Tuple[int, int, float]:
        """
        Get (view_index, timestamp_index, yaw_angle) for a frame index.

        With VIEW-FIRST ordering:
        - idx // n_timestamps = view index
        - idx % n_timestamps = timestamp index
        """
        v = idx // self.n_timestamps
        t = idx % self.n_timestamps
        yaw = self.yaw_angles[v]
        return v, t, yaw

    def _extract_all_frames(self):
        """Extract frames with motion detection."""
        import tempfile
        import shutil

        device = torch.device('cuda')
        temp_dir = tempfile.mkdtemp()
        front_path = os.path.join(temp_dir, 'front.mp4')
        back_path = os.path.join(temp_dir, 'back.mp4')

        try:
            print("Extracting perspectives with motion detection...")

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

            print("  Streams ready, analyzing motion...")

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

            extractor = GPUPerspectiveExtractor(
                fisheye_size=(self.decode_size, self.decode_size),
                output_size=self.output_size,
                output_fov=self.output_fov,
                native_size=self.native_width
            )

            frames_per_timestamp = self.input_fps / self.timestamps_per_second
            input_frame_count = 0
            timestamp_count = 0

            # Temporary storage for frames with motion
            frames_with_motion = []  # List of [views] arrays
            motion_scores = []

            prev_front_t = None
            prev_back_t = None

            def to_tensor(frame):
                ptr = frame.GetPtrToPlane(0)
                shape = frame.shape
                size = int(np.prod(shape))
                buf = (ctypes.c_uint8 * size).from_address(ptr)
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(shape).copy()
                t = torch.from_numpy(arr).to(device).float()
                return t.permute(2, 0, 1).unsqueeze(0)

            while timestamp_count < self.max_timestamps:
                front_batch = front_decoder.get_batch_frames(1)
                back_batch = back_decoder.get_batch_frames(1)

                if front_batch is None or back_batch is None:
                    break
                if len(front_batch) == 0 or len(back_batch) == 0:
                    break

                if input_frame_count >= timestamp_count * frames_per_timestamp:
                    front_t = to_tensor(front_batch[0])
                    back_t = to_tensor(back_batch[0])

                    # Check for motion
                    has_motion = True
                    motion_score = 0.0

                    if prev_front_t is not None:
                        motion_score = compute_motion_score(
                            front_t, back_t, prev_front_t, prev_back_t,
                            window_size=self.motion_window
                        )
                        has_motion = motion_score >= self.motion_threshold

                    if has_motion:
                        # Extract all views for this timestamp
                        views = []
                        for view_idx, yaw in enumerate(self.yaw_angles):
                            perspective = extractor.extract(front_t, back_t, yaw)
                            img = perspective.squeeze(0).permute(1, 2, 0)
                            img = img.clamp(0, 255).cpu().numpy() / 255.0
                            views.append(img.astype(self.dtype))

                        frames_with_motion.append(views)
                        self.motion_timestamps.append(timestamp_count)
                        motion_scores.append(motion_score)

                        # Update previous frame reference
                        prev_front_t = front_t.clone()
                        prev_back_t = back_t.clone()
                    else:
                        motion_scores.append(motion_score)

                    timestamp_count += 1
                    if timestamp_count % 50 == 0:
                        n_kept = len(frames_with_motion)
                        print(f"    Processed {timestamp_count}/{self.max_timestamps} timestamps, "
                              f"kept {n_kept} with motion ({100*n_kept/timestamp_count:.1f}%)")

                input_frame_count += 1

            # Convert to numpy array
            self.n_timestamps = len(frames_with_motion)
            self.total_frames = self.n_timestamps * self.views_per_timestamp

            if self.n_timestamps > 0:
                # frames_buffer[view, timestamp, h, w, 3]
                h, w = self.output_size
                self.frames_buffer = np.zeros(
                    (self.views_per_timestamp, self.n_timestamps, h, w, 3),
                    dtype=self.dtype
                )
                for t_idx, views in enumerate(frames_with_motion):
                    for v_idx, img in enumerate(views):
                        self.frames_buffer[v_idx, t_idx] = img

            skipped = timestamp_count - self.n_timestamps
            print(f"  Extraction complete:")
            print(f"    Total timestamps: {timestamp_count}")
            print(f"    Kept (with motion): {self.n_timestamps}")
            print(f"    Skipped (no motion): {skipped} ({100*skipped/max(1,timestamp_count):.1f}%)")
            print(f"    Total frames: {self.total_frames}")

            self.extraction_done = True

        except Exception as e:
            print(f"Extraction error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def __len__(self):
        if self.frames_buffer is None:
            self._extract_all_frames()
        return self.total_frames

    def __getitem__(self, idx):
        # Extract all frames on first access
        if self.frames_buffer is None:
            self._extract_all_frames()

        if not self.extraction_done or self.n_timestamps == 0:
            raise RuntimeError("Frame extraction failed or no motion detected")

        # View-first ordering: v = idx // n_timestamps, t = idx % n_timestamps
        v, t, yaw = self.get_frame_info(idx)
        img = self.frames_buffer[v, t]

        # Get original timestamp
        orig_t = self.motion_timestamps[t]
        timestamp = self.start_time + orig_t / self.timestamps_per_second

        self.timestamps.append(timestamp)
        return timestamp, img

    def get_timestamp(self, idx):
        if idx < len(self.timestamps):
            return self.timestamps[idx]
        v, t, _ = self.get_frame_info(idx)
        if t < len(self.motion_timestamps):
            orig_t = self.motion_timestamps[t]
            return self.start_time + orig_t / self.timestamps_per_second
        return 0.0

    def read_img(self, idx):
        _, img = self.__getitem__(idx)
        return (img * 255).astype(np.uint8)

    def get_image(self, idx):
        _, img = self.__getitem__(idx)
        return img

    def get_img_shape(self):
        return (384, 512), self.output_size

    def subsample(self, subsample):
        pass

    def has_calib(self):
        return False

    def stop(self):
        pass


def parse_insta360_multiview_dataset(dataset_path: str) -> Insta360MultiViewDataset:
    """
    Parse dataset path with optional parameters.

    Format: insta360mv:/path/to/video.insv?start=10&duration=60&views=10&fps=5&motion_thresh=5

    Parameters:
        start: Start time in seconds (default: 0)
        duration: Duration in seconds (default: 60)
        fov: Output FOV in degrees (default: 90)
        views: Views per timestamp (default: 10)
        fps: Timestamps per second (default: 5)
        motion_thresh: Motion threshold (default: 5.0, 0 to disable)
        motion_window: Motion detection window size (default: 64)
    """
    if dataset_path.startswith('insta360mv:'):
        dataset_path = dataset_path[11:]

    if '?' in dataset_path:
        path, query = dataset_path.split('?', 1)
        params = parse_qs(query)
    else:
        path = dataset_path
        params = {}

    start_time = float(params.get('start', [0])[0])
    duration = float(params.get('duration', [60])[0])
    output_fov = float(params.get('fov', [90])[0])
    views = int(params.get('views', [10])[0])
    fps = float(params.get('fps', [5])[0])
    motion_thresh = float(params.get('motion_thresh', [5.0])[0])
    motion_window = int(params.get('motion_window', [64])[0])

    return Insta360MultiViewDataset(
        video_path=path,
        start_time=start_time,
        duration=duration,
        output_fov=output_fov,
        views_per_timestamp=views,
        timestamps_per_second=fps,
        motion_threshold=motion_thresh,
        motion_window=motion_window
    )
