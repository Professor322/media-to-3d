import sys

import imageio
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(1, "./pycolmap/")
import pycolmap


def get_pix2cam(H, W, focal_x, focal_y):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="ij"
    )
    i = i.t()
    j = j.t()
    # Pixel coordinates -> Film coordinates -> Camera Coordinates. Y-axis pointing downwards.
    # In homogeneous coordinates Z is also pointing downwards
    # TODO: rewrite in terms of [i, j, 1] @ np.linalg.inv(cam2pix)
    return torch.stack(
        [(i - W / 2) / focal_x, -(j - H / 2) / focal_y, -torch.ones_like(i)], -1
    )


def get_cam2pix(focal_length_x, focal_length_y, principal_point_x, principal_point_y):
    """
    focal_lenght_x - focal length on x-axis
    focal_length_y - focal length on y-axix
    principal_point_x, principal_point_y - camera pinhole coordinates

    This function returns matrix that helps to transform image in the camera coordinates
    to pixel coordinates. Inverse of this matrix does the opposite, meaming pixel to camera coordinates
    """
    return torch.tensor(
        [
            [focal_length_x, 0, principal_point_x],
            [0, focal_length_y, principal_point_y],
            [0, 0, 1],
        ]
    )


def get_rays(ray_directions_cam, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Inputs:
        camera_coordinates: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # we only need to rotate to align with world coordinates, no need to tranlate.
    rays_d = ray_directions_cam @ c2w[:3, :3]  # (H, W, 3)

    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


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
        self.rays_directions_cam = get_pix2cam(
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
            ray_origins, ray_directions = get_rays(self.rays_directions_cam, world2cam)
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
