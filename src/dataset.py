import sys

import imageio
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(1, "/home/kolek/Edu/project/video-to-3d/src/pycolmap")
import pycolmap.pycolmap as pycolmap


def get_pixel_coords(H, W, focal):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()
    # NDC (-1, 1)
    return torch.stack(
        [(i - W / 2) / focal, -(j - H / 2) / focal, -torch.ones_like(i)], -1
    )


def get_rays(pixel_coords, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = pixel_coords @ c2w[:, :3].T  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


class NerfDataset(Dataset):
    """
    Class implements a dataloader for real world images
    Camera positions are estimated using colmap
    This is a blend of BlenderDataset and LLFF dataset
    """

    def __init__(self, data_path, image_width, image_height):
        self.scene_manager = pycolmap.SceneManager(data_path)
        self.scene_manager.load_cameras()
        self.scene_manager.load_images()
        self.image_resolution = [image_width, image_height]
        # assuming one camera is used for all pictures
        self.camera = self.scene_manager.cameras[1]
        # measured in pixels

        # matricies to convert from world coordinates to camera coordinates
        self.world2cams = []
        # image rgbs
        self.rgbs = []
        # rays casted from camera to each pixel
        self.rays_origins = []
        self.rays_directions = []
        self.transforms = transforms.ToTensor()
        self.near = 2.0
        self.far = 6.0
        self.pixel_coords = get_pixel_coords(
            self.image_resolution[0], self.image_resolution[1], self.camera.fx
        )

        # create world2cam matricies for images
        bottom = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        for img in self.scene_manager.images:
            translation = torch.from_numpy(img.C()).reshape(3, 1)
            rotation = torch.from_numpy(img.R())
            world2cam = torch.concatenate(
                [torch.concatenate([rotation, translation], dim=1), bottom], dim=0
            )
            self.world2cams.append(world2cam)
            # get image rgbs
            rgb = Image.open(self.scene_manager.image_path + img.name)
            if (
                rgb.width != self.image_resolution[0]
                or rgb.height != self.image_resolution[1]
            ):
                rgb = rgb.resize(self.image_resolution, Image.Resampling.LANCZOS)
            rgb = self.transforms(rgb)
            self.rgbs.append(rgb)
            ray_origins, ray_directions = get_rays(self.pixel_coords, world2cam)
            self.rays_origins.append(ray_origins)
            self.rays_directions.append(ray_directions)

    def __getitem__(self, index):
        return {
            "rgb": self.rgbs[index],
            "near": self.near,
            "far": self.far,
            "ray_origins": self.rays_origins[index],
            "ray_directions": self.rays_directions[index],
        }

    def __len__(self):
        return len(self.rgbs)
