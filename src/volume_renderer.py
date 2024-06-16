import torch
import torch.nn as nn

import ray_utils


class VolumeRenderer(nn.Module):
    """
    Module to render 2D view of our 3D model
    from the particular angle
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, radiance_field, ray_directions, ray_depth_samples, include_weights, device
    ):
        """
        radiance_field = (batch_size, n_rays, n_samples, 4) rgb, sigma
        ray_directions = (batch_size, n_rays, n_samples, 3) - direction of the ray
        ray_depth_samples = (batch_size, n_rays, n_samples) - depth of the ray directions

        Forward path of a renderer takes output of NeRF MLP and a ray along which
        we need to create a view.
        Output of the NeRF model is a colors along a ray and a sigma (color density)
        \gamma = t_{i + 1} - t_{i} - distance between adjacent samples
        \alpha = 1 - exp(-\sigma_{i} * \gamma_{i}) (alpha compositing value)
        C_{i}(r_{i}) = \sum(T_{i}(\alpha_{i}) * c_{i})  sum from t_near to t_far \\ equation (3)
        T_{i}(\alpha) = exp(-sum(\alpha))  sum from t_near to i
        """

        gammas = torch.cat(
            (
                ray_depth_samples[..., 1:] - ray_depth_samples[..., :-1],
                # last sample has no further samples, so add really big value here
                torch.tensor([1e10], device=device).expand(
                    ray_depth_samples[..., :1].shape
                ),
            ),
            dim=-1,
        )  # (N_rays, N_samples_)
        gammas = gammas * ray_directions[..., None, :].norm(p=2, dim=-1)
        rgb = radiance_field[..., :3]
        sigmas = radiance_field[..., 3]
        alpha = 1.0 - torch.exp(-sigmas * gammas)
        T = ray_utils.cumprod_exclusive(1.0 - alpha + 1e-10)

        weights = T * alpha
        # sum along the rays
        out_rgb = torch.sum(weights[..., None] * rgb, dim=-2)
        if include_weights == True:
            return [out_rgb, weights]
        else:
            return out_rgb
