import torch


class RaySampler(torch.nn.Module):
    """
    Implements stratified inverse depth sampling.
    https://arxiv.org/pdf/2003.08934
    page 6 (randomized offset in the bin)
    page 17 (inverse depth)
    """

    def __init__(self, num_samples):
        super(RaySampler, self).__init__()
        self.num_samples = num_samples

        # Ray sample count
        self.point_intervals = torch.linspace(
            0.0,
            1.0,
            self.num_samples,
            requires_grad=False,
        )

    def forward(self, ray_count, near, far, device):
        """
        Inputs:
            ray_count: int, number of rays in input ray chunk
            near: float or array of shape [BatchSize]. Nearest distance for a ray.
            far: float or array of shape [BatchSize]. Farthest distance for a ray.
        Outputs:
            point_intervals: (ray_count, self.num_samples) : depths of the sampled points along the ray
        """
        self.point_intervals = self.point_intervals.to(device)

        near, far = near * torch.ones_like(
            torch.empty(ray_count, 1, device=device)
        ), far * torch.ones_like(torch.empty(ray_count, 1, device=device))

        # The closer to the camera the more samples
        point_intervals = 1.0 / (
            1.0 / near * (1.0 - self.point_intervals) + 1.0 / far * self.point_intervals
        )

        # Get intervals between samples.
        mids = 0.5 * (point_intervals[..., 1:] + point_intervals[..., :-1])
        upper = torch.cat((mids, point_intervals[..., -1:]), dim=-1)
        lower = torch.cat((point_intervals[..., :1], mids), dim=-1)

        # Stratified samples in those intervals.
        t_rand = torch.rand(
            point_intervals.shape, dtype=point_intervals.dtype, device=device
        )
        point_intervals = lower + (upper - lower) * t_rand

        return point_intervals
