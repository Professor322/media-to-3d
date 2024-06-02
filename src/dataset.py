import sys

import imageio
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import utils

sys.path.insert(1, "./pycolmap/")
import pycolmap


class NerfDatasetRealImages(Dataset):
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
        # ray directions in camera coordinates
        self.rays_directions_cam = utils.get_pix2cam(
            self.image_resolution[0],
            self.image_resolution[1],
            self.camera.fx,
            self.camera.fy,
        )
        # create world2cam matricies for images
        bottom = torch.tensor([0, 0, 0, 1]).reshape(1, 4)
        for _, img in self.scene_manager.images.items():
            translation = torch.from_numpy(img.C()).reshape(3, 1)
            rotation = torch.from_numpy(img.R())
            world2cam = torch.concatenate(
                [torch.concatenate([rotation, translation], dim=1), bottom], dim=0
            ).float()
            self.world2cams.append(world2cam)
            # get image rgbs
            rgb = Image.open(self.scene_manager.image_path + img.name)
            if [rgb.width, rgb.height] != self.image_resolution:
                rgb = rgb.resize(self.image_resolution, Image.Resampling.LANCZOS)
            rgb = self.transforms(rgb)
            self.rgbs.append(rgb)
            ray_origins, ray_directions = utils.get_rays(
                self.rays_directions_cam, world2cam
            )
            self.rays_origins.append(ray_origins.float())
            self.rays_directions.append(ray_directions.float())

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
