import cv2
import numpy as np
import torch
from tqdm import tqdm

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
    counter = 0
    for rays in tqdm(
        system.val_dataset.get_render_rays(fps, video_duration),
        total=fps * video_duration,
    ):
        if dataset_type == "real":
            ray_origins, ray_directions = (
                rays["ray_origins"].cuda(),
                rays["ray_directions"].cuda(),
            )
            near, far = rays["near"], rays["far"]
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
        cv2.imwrite(f"./test/frame_{counter}.jpg", frame)
        writer.write(frame)
        counter += 1
    writer.release()
    print(f"video saved in {output_path + video_filename}")
    return output_path + video_filename


def render(config):
    render_path = ""
    nerfsys = NerfSystem.load_from_checkpoint(
        config.render_checkpoint_path, delete_validation_imgs=False
    )
    nerfsys.eval()
    nerfsys.setup("")
    if config.render_type == "video" or config.render_type == 1:
        render_path = render_video(
            system=nerfsys,
            fps=config.fps,
            video_duration=config.video_duration,
            output_path=config.output_path,
        )
    elif config.render_type == "mesh" or config.render_type == 2:
        raise NotImplementedError("Mesh generation not yet supported")

    return render_path
