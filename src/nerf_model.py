import torch
import torch.nn as nn


class NerfModel(nn.Module):
    """
    This class implements MLP architecture of the Nerf model
    from https://arxiv.org/pdf/2003.08934 page 18
    """

    def __init__(self, input_size_ray=3, input_size_direction=3, net_width=256):
        super().__init__()
        self.input_size_ray = input_size_ray
        self.input_size_direction = input_size_direction
        self.net_width = net_width

        self.model_first_part = nn.Sequential(
            nn.Linear(input_size_ray, net_width), nn.ReLU(inplace=True)
        )
        for _ in range(3):
            self.model_first_part = nn.Sequential(
                self.model_first_part,
                nn.Linear(net_width, net_width),
                nn.ReLU(inplace=True),
            )

        self.model_second_part = nn.Sequential(
            nn.Linear(input_size_ray + net_width, net_width), nn.ReLU(inplace=True)
        )
        for _ in range(2):
            self.model_second_part = nn.Sequential(
                self.model_second_part,
                nn.Linear(net_width, net_width),
                nn.ReLU(inplace=True),
            )

        self.model_second_part = nn.Sequential(
            self.model_second_part, nn.Linear(net_width, net_width)
        )

        self.model_sigma = nn.Linear(net_width + input_size_direction, net_width // 2 + 1)
        self.model_color = nn.Sequential(
            nn.ReLU(), nn.Linear(net_width // 2, 3), nn.Sigmoid()
        )
        self.float()

    def forward(self, ray, directions):
        """
        ray - (batch_size, n_rays, n_samples, dim_rays)
        directions - (batch_size, n_rays, n_samples, dim_directions)
        """
        out = self.model_first_part(ray)
        # prepare input for the second part ray + out
        out = torch.cat([ray, out], axis=-1)
        out = self.model_second_part(out)
        # prepare input for sigmas and colors out + directions
        out = torch.cat([directions, out], axis=-1)
        out = self.model_sigma(out)
        # extract sigmas and out for colors
        sigma, out = torch.split(out, [1, self.net_width // 2], dim=-1)

        color = self.model_color(out)

        # output dimensions (batch_size, n_rays, n_samples, 4)
        return torch.cat([sigma, color], axis=-1)
