import argparse
import sys

import cv2
import lightning as L
import numpy as np
import torch
from tqdm import tqdm

# this is ugly but it works
sys.path.insert(1, "./src")
from nerf_system import NerfSystem


def render_video(system=None, fps=30, video_duration=5, output_path=""):
    dataset_type = system.dataset_type
    W, H = system.image_resolution
    video_filename = "/render.mp4"
    writer = cv2.VideoWriter(
        output_path + video_filename,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (H, W),
    )
    for rays in tqdm(
        system.val_dataset.get_render_rays(fps, video_duration),
        total=fps * video_duration,
    ):
        if dataset_type == "real":
            pass
        elif dataset_type == "blender":
            rays = rays.cuda()
            ray_origins, ray_directions = (
                rays[:, 0:3],
                rays[:, 3:6],
            )
            near, far = rays[:, 6:7][0].item(), rays[:, 7:8][0].item()
        with torch.no_grad():
            frame = system(ray_origins, ray_directions, near, far)
        frame = np.clip(frame.cpu().numpy(), 0, 1) * 255
        frame = frame.reshape(H, W, 3).astype("uint8")
        writer.write(frame)
    writer.release()
    print(f"video saved in {output_path + video_filename}")


def render(config):
    nerfsys = NerfSystem.load_from_checkpoint(
        config.checkpoint_path, delete_validation_imgs=False
    )
    nerfsys.eval()
    nerfsys.setup("")
    if config.render_type == "video":
        render_video(
            system=nerfsys,
            fps=config.fps,
            video_duration=config.video_duration,
            output_path=config.output_path,
        )
    elif config.render_type == "mesh":
        raise NotImplementedError("Mesh generation not yet supported")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Checkpoint used to load the model",
    )
    parser.add_argument(
        "--render_type",
        type=str,
        default="video",
        choices={"video", "mesh"},
        help="Type of render to do",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per seconds if video is chosen as render type",
    )
    parser.add_argument(
        "--video_duration", type=int, default=5, help="How long the video to render"
    )
    parser.add_argument(
        "--output_path", type=str, default="", help="Path to save resulting render"
    )
    config = parser.parse_args()
    render(config)


if __name__ == "__main__":
    main()
