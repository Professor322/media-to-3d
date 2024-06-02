import lightning as L
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
        train_dataloader_path,
        batch_size,
        val_dataloder_path="",
    ):
        super().__init__()
        self.model = nerf_model.NerfModel(input_size_ray, input_size_direction)
        self.ray_sampler = ray_sampler.RaySampler(num_samples=n_ray_samples)
        self.volume_renderer = volume_renderer.VolumeRenderer()
        self.loss = torch.nn.MSELoss(reduction="mean")
        self.image_resolution = [image_width, image_height]
        self.train_dataloader_path = train_dataloader_path + "/sparse/0"
        self.val_dataloader_path = val_dataloder_path
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
        return rgb_out.reshape(
            batch_size, 3, self.image_resolution[0], self.image_resolution[1]
        )

    def setup(self, stage):
        self.train_dataset = dataset.NerfDatasetRealImages(
            data_path=self.train_dataloader_path,
            image_width=self.image_resolution[0],
            image_height=self.image_resolution[1],
        )
        # self.val_dataset = dataset.NerfDatasetRealImages(split="val", **kwargs)

    def training_step(self, batch, batch_nb):
        # near and far are the same for all elements in the batch
        near = batch["near"][0].item()
        far = batch["near"][0].item()
        results = self.forward(batch["ray_origins"], batch["ray_directions"], near, far)
        loss = self.loss(results, batch["rgb"])
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
