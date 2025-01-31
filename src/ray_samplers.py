import torch


class RaySamplerLinearInDisparity(torch.nn.Module):
    """
    Implements stratified inverse depth sampling.
    https://arxiv.org/pdf/2003.08934
    page 6 (randomized offset in the bin)
    page 17 (inverse depth)
    """

    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

        # Ray sample count
        self.point_intervals = torch.linspace(
            0.0,
            1.0,
            self.num_samples,
            requires_grad=False,
        )
        # avoid division by zero in case of near or far plane are 0
        self.eps = 1e-5

    def forward(self, ray_count, near, far, stratified, device):
        """
        Inputs:
            ray_count: int, number of rays in input ray chunk
            near: float or array of shape [BatchSize]. Nearest distance for a ray.
            far: float or array of shape [BatchSize]. Farthest distance for a ray.
        Outputs:
            point_intervals: (ray_count, self.num_samples) : depths of the sampled points along the ray
        """
        near += self.eps
        far += self.eps
        self.point_intervals = self.point_intervals.to(device)

        near, far = near * torch.ones_like(
            torch.empty(ray_count, 1, device=device)
        ), far * torch.ones_like(torch.empty(ray_count, 1, device=device))

        # The closer to the camera the more samples
        point_intervals = 1.0 / (
            1.0 / near * (1.0 - self.point_intervals) + 1.0 / far * self.point_intervals
        )

        # stratification should only be used for training
        if stratified:
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


class RaySamplerPDF(torch.nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.eps = 1e-5

    def forward(self, weights, stratified):
        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(self.eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
