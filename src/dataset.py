import json
import os
import sys

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
    Class implements a dataset for real world images
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
        print(self.camera.cx, self.camera.cy)

        # matricies to convert from world coordinates to camera coordinates
        self.c2w = []
        # image rgbs
        self.rgbs = []
        # rays casted from camera to each pixel
        self.rays_origins = []
        self.rays_directions = []
        self.transforms = transforms.ToTensor()
        self.near = 2.0
        self.far = 6.0
        self.image_names = []
        # matrix to translate from pixel to camera coordinates
        # self.pix2cam = utils.get_pix2cam(
        #    self.camera.fx,
        #    self.camera.fy,
        #    self.camera.cx,
        #    self.camera.cy,
        # )
        self.pix2cam = utils.get_pix2cam(
            self.camera.fx,
            self.camera.fy,
            image_width * 0.5,
            image_height * 0.5,
        )
        bottom = torch.tensor([0, 0, 0, 1]).reshape(1, 4)
        for _, img in self.scene_manager.images.items():
            translation = torch.from_numpy(img.tvec).reshape(3, 1)
            rotation = torch.from_numpy(img.R())
            world2cam = torch.concatenate(
                [torch.concatenate([rotation, translation], dim=1), bottom], dim=0
            ).float()
            cam2world = torch.linalg.inv(world2cam)
            self.c2w.append(cam2world)
            # get image rgbs
            rgb = Image.open(self.scene_manager.image_path + img.name)
            if [rgb.width, rgb.height] != self.image_resolution:
                rgb = rgb.resize(self.image_resolution, Image.Resampling.LANCZOS)
            rgb = self.transforms(rgb)
            self.rgbs.append(rgb)
            ray_origins, ray_directions = utils.get_rays(
                self.image_resolution[0],
                self.image_resolution[1],
                self.pix2cam,
                cam2world,
            )
            self.rays_origins.append(ray_origins.float())
            self.rays_directions.append(ray_directions.float())
            self.image_names.append(img.name)

    def __getitem__(self, index):
        return {
            "rgb": self.rgbs[index],
            "near": self.near,
            "far": self.far,
            "ray_origins": self.rays_origins[index],
            "ray_directions": self.rays_directions[index],
            "image_name": self.image_names[index],
        }

    def __len__(self):
        return len(self.rgbs)


def get_ray_direction(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        ray directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="ij"
    )
    i = i.t()
    j = j.t()
    # blender x -y, -z (only works with these signs)
    return torch.stack(
        [(i - W / 2) / focal, -(j - H / 2) / focal, -torch.ones_like(i)], -1
    )


def get_rays_with_dir(directions, c2w):
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
    # Why Transposed matrix ?
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


class BlenderDataset(Dataset):
    def __init__(self, root_dir, split="train", img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], "image width must equal image height!"
        self.img_wh = img_wh

        self.transform = transforms.ToTensor()
        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r") as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = (
            0.5 * 800 / np.tan(0.5 * self.meta["camera_angle_x"])
        )  # original focal length
        # when W=800

        self.focal *= (
            self.img_wh[0] / 800
        )  # modify focal length to match size self.img_wh

        # ray directions for all pixels in camera coordinates, same for all images (same H, W, focal)
        self.dir_cam = get_ray_direction(h, w, self.focal)  # (h, w, 3)

        if self.split == "train":  # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for frame in self.meta["frames"]:
                pose = np.array(frame["transform_matrix"])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA -> (640000, 4)
                img = img[:, :3] * img[:, -1:] + (
                    1 - img[:, -1:]
                )  # blend A to RGB -> (h*w, 3)
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays_with_dir(self.dir_cam, c2w)  # both (h*w, 3)

                self.all_rays += [
                    torch.cat(
                        [
                            rays_o,
                            rays_d,
                            self.near * torch.ones_like(rays_o[:, :1]),
                            self.far * torch.ones_like(rays_o[:, :1]),
                        ],
                        1,
                    )
                ]  # (h*w, 8)
            """
            flatten all rays/rgb tensor
                * self.all_rgbs[idx] -> (r,g,b)
                * self.all_rays[idx] -> (ox,oy,oz,dx,dy,dz,near,far)
            """
            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 8) -> (100x800x800, 8)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3) -> (100x800x800, 3)

    def __len__(self):
        if self.split == "train":
            return len(self.all_rays)
        if self.split == "val":
            return 8  # only validate 8 images (to support <=8 gpus)
        return len(self.meta["frames"])

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}

        else:  # create data for each image separately
            frame = self.meta["frames"][idx]
            c2w = torch.FloatTensor(frame["transform_matrix"])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, H, W)
            valid_mask = (img[-1] > 0).flatten()  # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0)  # (H*W, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            rays_o, rays_d = get_rays_with_dir(self.dir_cam, c2w)

            rays = torch.cat(
                [
                    rays_o,
                    rays_d,
                    self.near * torch.ones_like(rays_o[:, :1]),
                    self.far * torch.ones_like(rays_o[:, :1]),
                ],
                1,
            )  # (H*W, 8)

            sample = {"rays": rays, "rgbs": img, "c2w": c2w, "valid_mask": valid_mask}

        return sample
