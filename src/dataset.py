import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import ray_utils
import render_utils

sys.path.insert(1, "./src/pycolmap/")
import pycolmap


class NerfDatasetRealImages(Dataset):
    """
    Class implements a dataset for real world images
    Camera positions are estimated using colmap
    This is a blend of BlenderDataset and LLFF dataset
    """

    def __init__(self, data_path, downscale_factor=1, split="train"):
        self.data_path = data_path
        self.split = split
        self.downscale_factor = downscale_factor
        # work around if we end up having multiple camera positions estimations
        max_images = 0
        camera_position_estimations_idx = max_camera_position_estimation_idx = 0
        while os.path.exists(data_path + f"/sparse/{camera_position_estimations_idx}"):
            self.scene_manager = pycolmap.SceneManager(
                data_path + f"/sparse/{camera_position_estimations_idx}"
            )
            self.scene_manager.load_images()
            if len(self.scene_manager.images.items()) > max_images:
                max_camera_position_estimation_idx = camera_position_estimations_idx
                max_images = len(self.scene_manager.images.items())
            camera_position_estimations_idx += 1

        print(f"Reconstruction {max_camera_position_estimation_idx} chosen")
        # reload reconstruction with max images in it
        self.scene_manager = pycolmap.SceneManager(
            data_path + f"/sparse/{max_camera_position_estimation_idx}"
        )
        self.scene_manager.load_images()

        self.scene_manager.load_cameras()
        # assuming one camera is used for all pictures
        self.camera = self.scene_manager.cameras[1]

        # image rgbs
        self.rgbs = []
        # rays casted from camera to each pixel
        self.rays_origins = []
        self.rays_directions = []
        self.transforms = transforms.ToTensor()
        # TODO: estimate near and far planes for each image,
        # currently does not work properly
        self.near = 2.0
        self.far = 3.0
        self.image_names = []

        # matrix to translate from pixel to camera coordinates
        self.pix2cam = ray_utils.get_pix2cam(
            self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy
        )
        self.distortion_params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        if self.camera.camera_type == 2:  # simple radial
            self.distortion_params["k1"] = self.camera.k1
        elif self.camera.camera_type == 4:  # opencv
            self.distortion_params["k1"] = self.camera.k1
            self.distortion_params["k2"] = self.camera.k2
            self.distortion_params["p1"] = self.camera.p1
            self.distortion_params["p2"] = self.camera.p2

        # initialized when first image read
        self.image_resolution = None
        world2cams = []
        bottom = torch.tensor([0, 0, 0, 1]).reshape(1, 4)
        for _, img in self.scene_manager.images.items():
            translation = torch.from_numpy(img.tvec).reshape(3, 1)
            rotation = torch.from_numpy(img.R())
            world2cam = torch.concatenate(
                [torch.concatenate([rotation, translation], dim=1), bottom], dim=0
            ).float()
            world2cams.append(world2cam)
            # get image rgbs
            rgb = Image.open(self.data_path + f"/images/" + img.name).convert("RGB")
            if [rgb.width, rgb.height] != self.image_resolution:
                if self.image_resolution == None:
                    new_image_width = int(rgb.width / self.downscale_factor)
                    new_image_height = int(rgb.height / self.downscale_factor)
                    self.image_resolution = [new_image_width, new_image_height]
                    print(f"Scaled image resolution: {self.image_resolution}")
                    self.pix2cam = self.pix2cam @ torch.diag(
                        torch.tensor([self.downscale_factor, self.downscale_factor, 1.0])
                    )
                rgb = rgb.resize(self.image_resolution, Image.Resampling.LANCZOS)

            rgb = self.transforms(rgb)
            self.rgbs.append(rgb)
            self.image_names.append(img.name)

        world2cams = torch.stack(world2cams)
        cam2worlds = torch.linalg.inv(world2cams)
        self.poses = cam2worlds[..., :3, :4]
        # fit poses into unit cube, from multinerf
        self.poses = ray_utils.transform_poses_pca(self.poses)
        for pose in self.poses:
            ray_origins, ray_directions = ray_utils.get_rays(
                self.image_resolution[0],
                self.image_resolution[1],
                self.pix2cam,
                pose,
                self.distortion_params,
            )
            self.rays_origins.append(ray_origins.float())
            self.rays_directions.append(ray_directions.float())
        self.rays_origins = torch.stack(self.rays_origins).reshape(-1, 3)
        self.rays_directions = torch.stack(self.rays_directions).reshape(-1, 3)
        self.rgbs = torch.stack(self.rgbs).permute(0, 2, 3, 1).reshape(-1, 3)

    def __getitem__(self, index):
        if self.split == "train":
            return {
                "rgb": self.rgbs[index],
                "near": self.near,
                "far": self.far,
                "ray_origins": self.rays_origins[index],
                "ray_directions": self.rays_directions[index],
            }
        else:
            num_samples = self.image_resolution[0] * self.image_resolution[1]
            sample_begin = index * num_samples
            sample_end = (index + 1) * num_samples
            return {
                "rgb": self.rgbs[sample_begin:sample_end],
                "near": self.near,
                "far": self.far,
                "ray_origins": self.rays_origins[sample_begin:sample_end],
                "ray_directions": self.rays_directions[sample_begin:sample_end],
                "image_name": self.image_names[index],
            }

    def __len__(self):
        return len(self.rgbs) if self.split == "train" else 1

    def get_render_rays(self, duration=5, fps=30):
        self.render_poses = render_utils.generate_ellipse_path(
            self.poses, n_frames=duration * fps
        )
        for render_pose in self.render_poses:
            ray_origins, ray_directions = ray_utils.get_rays(
                self.image_resolution[0],
                self.image_resolution[1],
                self.pix2cam,
                render_pose,
                self.distortion_params,
            )
            rays = {
                "ray_origins": ray_origins,
                "ray_directions": ray_directions,
                "near": self.near,
                "far": self.far,
            }

            yield rays


# initial code was copied, but has been modified for rendering needs


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
        self.dir_cam = ray_utils.get_ray_direction(h, w, self.focal)  # (h, w, 3)

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

                rays_o, rays_d = ray_utils.get_rays_with_dir(
                    self.dir_cam, c2w
                )  # both (h*w, 3)

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

            rays_o, rays_d = ray_utils.get_rays_with_dir(self.dir_cam, c2w)

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

    def get_render_rays(self, duration=5, fps=30):
        self.render_poses = render_utils.create_spherical_poses(num_poses=duration * fps)
        # self.render_poses = torch.stack(self.render_poses)
        for render_pose in self.render_poses:
            rays_o, rays_d = ray_utils.get_rays_with_dir(
                self.dir_cam, render_pose[:3, :4]
            )

            rays = torch.cat(
                [
                    rays_o,
                    rays_d,
                    self.near * torch.ones_like(rays_o[:, :1]),
                    self.far * torch.ones_like(rays_o[:, :1]),
                ],
                1,
            )  # (H*W, 8)
            yield rays
