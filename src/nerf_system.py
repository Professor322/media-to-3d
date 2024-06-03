import lightning as L
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import dataset
import nerf_model
import ray_sampler
import utils
import volume_renderer


class NerfSystem(L.LightningModule):
    def __init__(
        self,
        input_size_ray,
        input_size_direction,
        n_ray_samples,
        image_width,
        image_height,
        dataset_type,
        train_dataset_path,
        batch_size,
        val_dataset_path="",
    ):
        super().__init__()
        self.model = nerf_model.NerfModel(input_size_ray, input_size_direction)
        self.ray_sampler = ray_sampler.RaySampler(num_samples=n_ray_samples)
        self.volume_renderer = volume_renderer.VolumeRenderer()
        self.loss = torch.nn.MSELoss(reduction="mean")
        self.image_resolution = [image_width, image_height]
        self.dataset_type = dataset_type
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        if self.dataset_type == "real":
            self.train_dataset_path += "/sparse/0"
            self.val_dataset_path += "sparse/0"

        self.batch_size = batch_size

    def forward(self, ray_origins, ray_directions, near, far):
        batch_size = ray_directions.shape[0]
        ray_count = ray_directions.shape[1]

        ray_depth_values = self.ray_sampler(
            ray_count, near=near, far=far, device=self.device
        ).float()
        ray_points = utils.intervals_to_ray_points(
            ray_depth_values, ray_directions, ray_origins
        )

        # expand ray_directions to match ray point size to feed into MLP
        expanded_ray_directions = (
            ray_directions[..., None, :].expand_as(ray_points).float()
        )
        radiance_field = self.model(ray_points, expanded_ray_directions)
        # render volume to rgb
        rgb_out = self.volume_renderer(
            radiance_field, ray_directions, ray_depth_values, self.device
        )
        # (batch_size, num_rays, 3) -> (batch_size, 3, H, W)
        return rgb_out.permute(0, 2, 1).reshape(
            batch_size, 3, self.image_resolution[0], self.image_resolution[1]
        )

    def setup(self, stage):
        if self.dataset_type == "real":
            self.train_dataset = dataset.NerfDatasetRealImages(
                data_path=self.train_dataset_path,
                image_width=self.image_resolution[0],
                image_height=self.image_resolution[1],
            )
            # self.val_dataset = dataset.NerfDatasetRealImages(split="val", **kwargs)
        elif self.dataset_type == "blender":
            self.train_dataset = dataset.BlenderDataset(
                self.train_dataset_path, split="train", img_wh=self.image_resolution
            )
            self.val_dataset = dataset.BlenderDataset(
                self.train_dataset_path, split="val", img_wh=self.image_resolution
            )

    def training_step(self, batch, batch_nb):
        # near and far are the same for all elements in the batch
        if self.dataset_type == "real":
            near = batch["near"][0].item()
            far = batch["near"][0].item()
            ray_origins, ray_directions = batch["ray_origins"], batch["ray_directions"]
            rgbs = batch["rgb"]
        elif self.dataset_type == "blender":
            ray_origins, ray_directions = (
                batch["rays"][None, :, 0:3],
                batch["rays"][None, :, 3:6],
            )
            near, far = batch["rays"][:, 6:7][0].item(), batch["rays"][:, 7:8][0].item()
            rgbs = (
                batch["rgbs"]
                .permute(1, 0)
                .reshape(1, 3, self.image_resolution[0], self.image_resolution[1])
            )

        results = self.forward(ray_origins, ray_directions, near, far)
        loss = self.loss(results, rgbs)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=5e-3, betas=(0.9, 0.999)
        )
        return [self.optimizer]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=4,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.Any:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,
            pin_memory=True,
        )

    def validation_step(self, batch, batch_nb):
        if self.dataset_type == "real":
            near = batch["near"][0].item()
            far = batch["near"][0].item()
            ray_origins, ray_directions = batch["ray_origins"], batch["ray_directions"]
            rgbs = batch["rgb"]
        elif self.dataset_type == "blender":
            ray_origins, ray_directions = batch["rays"][..., 0:3], batch["rays"][..., 3:6]
            near, far = (
                batch["rays"][..., 6:7][0, 0].item(),
                batch["rays"][..., 7:8][0, 0].item(),
            )
            rgbs = (
                batch["rgbs"]
                .permute(0, 2, 1)
                .reshape(1, 3, self.image_resolution[0], self.image_resolution[1])
            )
        results = self.forward(ray_origins, ray_directions, near, far)
        log = {"val_loss": self.loss(results, rgbs)}

        if batch_nb == 0:
            plt.subplot(121)
            plt.imshow(results[0].permute(1, 2, 0).cpu().numpy())
            plt.axis("off")
            plt.subplot(122)
            plt.imshow(rgbs[0].permute(1, 2, 0).cpu().numpy())
            plt.axis("off")
            plt.show()

        loss = self.loss(results, rgbs)
        log["val_psnr"] = loss
        return log