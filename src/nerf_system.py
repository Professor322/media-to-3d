import lightning as L
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import dataset
import nerf_model
import positional_encoder
import ray_samplers
import utils
import volume_renderer


class NerfSystem(L.LightningModule):
    def __init__(
        self,
        use_positional_encoding,
        use_hierarchical_sampling,
        input_size_ray,
        input_size_direction,
        n_ray_samples,
        dataset_type,
        train_dataset_path,
        batch_size,
        downscale_factor,
        val_dataset_path="",
    ):
        super().__init__()

        self.image_resolution = None
        self.use_hierarchical_sampling = use_hierarchical_sampling
        self.downscale_factor = downscale_factor
        self.automatic_optimization = self.use_hierarchical_sampling != True
        self.ray_sampler_in_linear_disparity = ray_samplers.RaySamplerLinearInDisparity(
            num_samples=n_ray_samples
        )
        self.ray_sampler_pdf = ray_samplers.RaySamplerPDF(num_samples=n_ray_samples)
        self.volume_renderer = volume_renderer.VolumeRenderer()
        self.loss = torch.nn.MSELoss(reduction="mean")
        self.dataset_type = dataset_type
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoder_ray_direction = self.positional_encoder_ray_points = None
        self.input_size_ray = input_size_ray
        self.input_size_direction = input_size_direction
        if self.use_positional_encoding:
            # \gamma(x) for direction L = 4 as in paper
            self.positional_encoder_ray_direction = positional_encoder.PositionalEncoder(
                input_channels=3, num_freqs=4
            )
            # \gamma(x) for sampled ray points L = 10 as in paper
            self.positional_encoder_ray_points = positional_encoder.PositionalEncoder(
                input_channels=3, num_freqs=10
            )
            # when we are using positional encoding input size will be expanded and original values will be ignored
            self.input_size_ray = self.positional_encoder_ray_points.output_channels
            self.input_size_direction = (
                self.positional_encoder_ray_direction.output_channels
            )

        print(f"Model inputs {self.input_size_ray, self.input_size_direction}")

        self.coarse_model = nerf_model.NerfModel(
            self.input_size_ray, self.input_size_direction
        )
        if self.use_hierarchical_sampling:
            self.fine_model = nerf_model.NerfModel(
                self.input_size_ray, self.input_size_direction
            )

        self.batch_size = batch_size

    def coarse_rgb(
        self, ray_count, near, far, ray_origins, ray_directions, do_stratification
    ):
        ray_depth_values = self.ray_sampler_in_linear_disparity(
            ray_count,
            near=near,
            far=far,
            stratified=do_stratification,
            device=self.device,
        ).float()
        ray_points = utils.intervals_to_ray_points(
            ray_depth_values, ray_directions, ray_origins
        )
        # expand ray_directions to match ray point size to feed into MLP
        expanded_ray_directions = (
            ray_directions[..., None, :].expand_as(ray_points).float()
        )
        if self.use_positional_encoding:
            ray_points = self.positional_encoder_ray_points(ray_points)
            expanded_ray_directions = self.positional_encoder_ray_direction(
                expanded_ray_directions
            )

        radiance_field = self.coarse_model(ray_points, expanded_ray_directions)
        # render volume to rgb
        return self.volume_renderer(
            radiance_field, ray_directions, ray_depth_values, True, self.device
        )

    def fine_rgb(
        self,
        ray_count,
        near,
        far,
        ray_origins,
        ray_directions,
        weights,
        do_stratification,
    ):
        ray_depth_values = self.ray_sampler_pdf(
            ray_count,
            near=near,
            far=far,
            stratified=do_stratification,
            weights=weights,
            device=self.device,
        ).float()
        ray_points = utils.intervals_to_ray_points(
            ray_depth_values, ray_directions, ray_origins
        )
        if self.use_positional_encoding:
            ray_points = self.positional_encoder_ray_points(ray_points)
            expanded_ray_directions = self.positional_encoder_ray_direction(
                expanded_ray_directions
            )

        radiance_field = self.fine_model(ray_points, expanded_ray_directions)
        return self.volume_renderer(
            radiance_field, ray_directions, ray_depth_values, False, self.device
        )

    def forward(self, ray_origins, ray_directions, near, far):
        batch_size = ray_directions.shape[0]
        ray_count = ray_directions.shape[1]
        # do_stratification = self.training == True
        do_stratification = True
        coarse_rgb, weights = self.coarse_rgb(
            ray_count, near, far, ray_origins, ray_directions, do_stratification
        )

        if self.use_hierarchical_sampling:
            fine_rgb = self.fine_rgb(
                ray_count,
                near,
                far,
                ray_origins,
                ray_directions,
                weights,
                do_stratification,
            )
            return [
                coarse_rgb.permute(0, 2, 1).reshape(
                    batch_size, 3, self.image_resolution[1], self.image_resolution[0]
                ),
                fine_rgb.permute(0, 2, 1).reshape(
                    batch_size, 3, self.image_resolution[1], self.image_resolution[0]
                ),
            ]
        # (batch_size, num_rays, 3) -> (batch_size, 3, W, H)
        return coarse_rgb.permute(0, 2, 1).reshape(
            batch_size, 3, self.image_resolution[1], self.image_resolution[0]
        )

    def setup(self, stage):
        if self.dataset_type == "real":
            self.train_dataset = dataset.NerfDatasetRealImages(
                data_path=self.train_dataset_path,
                downscale_factor=self.downscale_factor,
                split="train",
            )
            # shortcut, not ideal
            self.image_resolution = self.train_dataset.image_resolution
            # TODO:
            # we will use the same dataset as for training, but for validation
            # we should use unseen poses
            self.val_dataset = dataset.NerfDatasetRealImages(
                data_path=self.train_dataset_path,
                downscale_factor=self.downscale_factor,
                split="val",
            )
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
        out_opt = []

        self.optimizer_coarse = torch.optim.Adam(
            self.coarse_model.parameters(), lr=5e-3, betas=(0.9, 0.999)
        )
        out_opt.append(self.optimizer_coarse)

        if self.use_hierarchical_sampling:
            self.optimizer_fine = torch.optim.Adam(
                self.fine_model.parameters(), lr=5e-3, betas=(0.9, 0.999)
            )
            out_opt.append(self.optimizer_fine)
        return out_opt

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
