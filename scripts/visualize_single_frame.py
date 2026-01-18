"""
Extract and visualize all components of a single frame for QA:
- Front fisheye
- Back fisheye
- Equirectangular projection
- All 10 (or 12) perspective views

Usage:
    python scripts/visualize_single_frame.py /path/to/video.insv --time 100 --views 10 --output qa_frame
"""

import argparse
import os
import subprocess
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path

try:
    import PyNvVideoCodec as nvc
    HAS_PYNVVIDEOCODEC = True
except ImportError:
    HAS_PYNVVIDEOCODEC = False


class GPUPerspectiveExtractor:
    """GPU-resident fisheye to perspective converter."""

    def __init__(
        self,
        fisheye_size,
        output_size=(512, 512),
        output_fov=90.0,
        device='cuda',
        native_size=3840
    ):
        self.device = torch.device(device)
        self.fisheye_size = fisheye_size
        self.output_size = output_size
        self.output_fov = output_fov

        # Equidistant fisheye: r = f * theta
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

    def _compute_grid_for_fisheye(self, yaw_rad, for_front):
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

    def extract(self, front_tensor, back_tensor, yaw):
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


def fisheye_to_equirectangular(front_tensor, back_tensor, fisheye_size, output_size=(2048, 1024), device='cuda'):
    """Convert dual fisheye to equirectangular projection."""
    out_w, out_h = output_size
    fish_w, fish_h = fisheye_size

    # Equidistant fisheye focal length
    native_f = 1100.0
    f = native_f * (fisheye_size[0] / 3840)

    # Create equirectangular coordinates
    u = torch.linspace(0, out_w - 1, out_w, device=device)
    v = torch.linspace(0, out_h - 1, out_h, device=device)
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')

    # Convert to spherical coordinates
    # longitude: -pi to pi (left to right)
    # latitude: -pi/2 to pi/2 (bottom to top)
    lon = (u_grid / out_w) * 2 * np.pi - np.pi
    lat = (v_grid / out_h) * np.pi - np.pi / 2

    # Convert to 3D direction
    x = torch.cos(lat) * torch.sin(lon)
    y = torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)

    # Determine which fisheye to use (front: z > 0, back: z < 0)
    use_front = z >= 0

    # For each fisheye, compute the projection
    # Front fisheye looks along +Z
    # Back fisheye looks along -Z

    # Front fisheye projection
    r_xy_front = torch.sqrt(x**2 + y**2)
    theta_front = torch.atan2(r_xy_front, z)
    r_front = f * theta_front
    phi_front = torch.atan2(y, x)
    u_front = r_front * torch.cos(phi_front) + fish_w / 2
    v_front = r_front * torch.sin(phi_front) + fish_h / 2

    # Back fisheye projection (looking along -Z)
    r_xy_back = torch.sqrt(x**2 + y**2)
    theta_back = torch.atan2(r_xy_back, -z)
    r_back = f * theta_back
    phi_back = torch.atan2(y, -x)  # flip x for mirror
    u_back = r_back * torch.cos(phi_back) + fish_w / 2
    v_back = r_back * torch.sin(phi_back) + fish_h / 2

    # Normalize coordinates for grid_sample
    u_front_norm = 2.0 * u_front / (fish_w - 1) - 1.0
    v_front_norm = 2.0 * v_front / (fish_h - 1) - 1.0
    u_back_norm = 2.0 * u_back / (fish_w - 1) - 1.0
    v_back_norm = 2.0 * v_back / (fish_h - 1) - 1.0

    # Clip invalid projections
    valid_front = (theta_front < np.radians(100)) & use_front
    valid_back = (theta_back < np.radians(100)) & ~use_front

    u_front_norm = torch.where(valid_front, u_front_norm, torch.tensor(2.0, device=device))
    v_front_norm = torch.where(valid_front, v_front_norm, torch.tensor(2.0, device=device))
    u_back_norm = torch.where(valid_back, u_back_norm, torch.tensor(2.0, device=device))
    v_back_norm = torch.where(valid_back, v_back_norm, torch.tensor(2.0, device=device))

    grid_front = torch.stack([u_front_norm, v_front_norm], dim=-1).unsqueeze(0)
    grid_back = torch.stack([u_back_norm, v_back_norm], dim=-1).unsqueeze(0)

    # Sample from each fisheye
    sample_front = F.grid_sample(front_tensor, grid_front, mode='bilinear', padding_mode='zeros', align_corners=True)
    sample_back = F.grid_sample(back_tensor, grid_back, mode='bilinear', padding_mode='zeros', align_corners=True)

    # Blend based on which fisheye is valid
    use_front_mask = use_front.unsqueeze(0).unsqueeze(0).float()
    equirect = sample_front * use_front_mask + sample_back * (1 - use_front_mask)

    return equirect


def main():
    parser = argparse.ArgumentParser(description="Visualize single frame components")
    parser.add_argument("video_path", help="Path to .insv file")
    parser.add_argument("--time", type=float, default=100.0, help="Time in seconds to extract")
    parser.add_argument("--views", type=int, default=10, help="Number of perspective views")
    parser.add_argument("--output", type=str, default="qa_frame", help="Output directory name")
    parser.add_argument("--fov", type=float, default=90.0, help="Perspective FOV in degrees")
    args = parser.parse_args()

    device = torch.device('cuda')
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Extracting frame at t={args.time}s from {args.video_path}")

    # Probe video
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'stream=index,codec_type,width,height,r_frame_rate',
        '-of', 'csv=p=0', args.video_path
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

    native_width = video_streams[0]['width']
    native_height = video_streams[0]['height']
    fps_num, fps_den = map(int, video_streams[0]['fps'].split('/'))
    input_fps = fps_num / fps_den

    print(f"Video: {native_width}x{native_height} @ {input_fps} fps")

    # Extract single frame from each stream using ffmpeg
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    front_frame_path = os.path.join(temp_dir, 'front.png')
    back_frame_path = os.path.join(temp_dir, 'back.png')

    try:
        # Extract front fisheye frame
        subprocess.run([
            'ffmpeg', '-y', '-ss', str(args.time), '-i', args.video_path,
            '-map', '0:v:0', '-frames:v', '1', front_frame_path
        ], capture_output=True, check=True)

        # Extract back fisheye frame
        subprocess.run([
            'ffmpeg', '-y', '-ss', str(args.time), '-i', args.video_path,
            '-map', '0:v:1', '-frames:v', '1', back_frame_path
        ], capture_output=True, check=True)

        # Load frames
        front_np = cv2.imread(front_frame_path)
        front_np = cv2.cvtColor(front_np, cv2.COLOR_BGR2RGB)
        back_np = cv2.imread(back_frame_path)
        back_np = cv2.cvtColor(back_np, cv2.COLOR_BGR2RGB)

        print(f"Loaded fisheye frames: {front_np.shape}")

        # Save fisheyes
        cv2.imwrite(str(output_dir / "01_front_fisheye.png"),
                    cv2.cvtColor(front_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / "02_back_fisheye.png"),
                    cv2.cvtColor(back_np, cv2.COLOR_RGB2BGR))
        print("  Saved: 01_front_fisheye.png, 02_back_fisheye.png")

        # Convert to tensors
        front_t = torch.from_numpy(front_np).to(device).float()
        front_t = front_t.permute(2, 0, 1).unsqueeze(0)
        back_t = torch.from_numpy(back_np).to(device).float()
        back_t = back_t.permute(2, 0, 1).unsqueeze(0)

        fisheye_size = (front_np.shape[1], front_np.shape[0])

        # Create equirectangular
        print("Creating equirectangular projection...")
        equirect = fisheye_to_equirectangular(front_t, back_t, fisheye_size, device=device)
        equirect_np = equirect.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imwrite(str(output_dir / "03_equirectangular.png"),
                    cv2.cvtColor(equirect_np, cv2.COLOR_RGB2BGR))
        print("  Saved: 03_equirectangular.png")

        # Create perspectives
        print(f"Creating {args.views} perspective views...")
        extractor = GPUPerspectiveExtractor(
            fisheye_size=fisheye_size,
            output_size=(512, 512),
            output_fov=args.fov,
            native_size=native_width
        )

        view_yaws = [i * (360.0 / args.views) for i in range(args.views)]

        for i, yaw in enumerate(view_yaws):
            perspective = extractor.extract(front_t, back_t, yaw)
            persp_np = perspective.squeeze(0).permute(1, 2, 0).cpu().numpy()
            persp_np = persp_np.clip(0, 255).astype(np.uint8)

            filename = f"04_perspective_{i:02d}_yaw{int(yaw):03d}.png"
            cv2.imwrite(str(output_dir / filename),
                        cv2.cvtColor(persp_np, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {filename}")

        # Create a combined view of all perspectives
        print("Creating combined perspectives grid...")
        grid_cols = min(5, args.views)
        grid_rows = (args.views + grid_cols - 1) // grid_cols

        cell_size = 256
        grid_img = np.zeros((grid_rows * cell_size, grid_cols * cell_size, 3), dtype=np.uint8)

        for i, yaw in enumerate(view_yaws):
            row = i // grid_cols
            col = i % grid_cols

            perspective = extractor.extract(front_t, back_t, yaw)
            persp_np = perspective.squeeze(0).permute(1, 2, 0).cpu().numpy()
            persp_np = persp_np.clip(0, 255).astype(np.uint8)

            # Resize for grid
            persp_small = cv2.resize(persp_np, (cell_size, cell_size))

            # Add label
            cv2.putText(persp_small, f"View {i}: {int(yaw)}°",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            grid_img[y1:y2, x1:x2] = persp_small

        cv2.imwrite(str(output_dir / "05_perspectives_grid.png"),
                    cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
        print("  Saved: 05_perspectives_grid.png")

        print(f"\nAll outputs saved to: {output_dir}/")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
