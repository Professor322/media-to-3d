import torch


def get_pixel_coordinates(H, W):
    return torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="ij"
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


def get_pix2cam(focal_x, focal_y, principal_point_x, principal_point_y):
    return torch.linalg.inv(
        get_cam2pix(focal_x, focal_y, principal_point_x, principal_point_y)
    )


def get_rays(H, W, pix2cam, cam2world):
    """
    H - height
    W - width
    pix2cam - translation matrix from pixels to camera coordinates (3, 3)
    cam2world - tranlsation matrix from camera coordinates to world coordinates (3, 4)
    """
    x, y = get_pixel_coordinates(H, W)

    def pixel_to_direction(x, y):
        # shoot ray through the middle of the pixel
        return torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1)

    # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
    cam2world = cam2world @ torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

    directions_camera_coords = pixel_to_direction(x, y) @ pix2cam
    # https://github.com/colmap/colmap/issues/1341
    directions_real_world_coords = directions_camera_coords @ cam2world[:3, :3].T
    directions_real_world_coords = directions_real_world_coords / torch.norm(
        directions_real_world_coords, dim=-1, keepdim=True
    )

    origins = cam2world[..., :3, -1].expand(directions_real_world_coords.shape)
    return origins.reshape(-1, 3), directions_camera_coords.reshape(-1, 3)


# def get_rays(pix2cma, c2w):
#    """
#    Get ray origin and normalized directions in world coordinate for all pixels in one image.
#    Inputs:
#        camera_coordinates: (H, W, 3) precomputed ray directions in camera coordinate
#        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
#    Outputs:
#        rays_o: (H*W, 3), the origin of the rays in world coordinate
#        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
#    """
#    # we only need to rotate to align with world coordinates, no need to tranlate.
#    rays_d = ray_directions_cam @ c2w[:3, :3]  # (H, W, 3)
#    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
#    # The origin of all rays is the camera origin in world coordinate
#    rays_o = c2w[:3, -1].expand(rays_d.shape)  # (H, W, 3)
#    rays_d = rays_d.view(-1, 3)
#    rays_o = rays_o.view(-1, 3)
#
#    return rays_o, rays_d


def intervals_to_ray_points(point_intervals, ray_directions, ray_origin):
    """Through depths of the sampled positions('point_intervals') and the starting point('ray_origin')
        and direction vector of the light('ray_directions'), calculate each point on the ray
    Args:
        point_intervals (torch.tensor): (ray_count, num_samples) : Depths of the sampled positions along the ray
        ray_directions (torch.tensor): (ray_count, 3)
        ray_origin (torch.tensor): (ray_count, 3)
    Return:
        ray_points(torch.tensor): Samples points along each ray: (ray_count, num_samples, 3)
    """
    return (
        ray_origin[..., None, :]
        + ray_directions[..., None, :] * point_intervals[..., :, None]
    )


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.
    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod
