import numpy as np
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


def get_rays(H, W, pix2cam, pose, distortion_params=None):
    """
    H - height
    W - width
    pix2cam - translation matrix from pixels to camera coordinates (3, 3)
    pose - tranlsation matrix from camera coordinates to world coordinates (3, 4)
    distortion_params
    """
    x, y = get_pixel_coordinates(H, W)

    def pixel_to_direction(x, y):
        return torch.stack([x, y, torch.ones_like(x)], dim=-1)

    mat_vec_mul = lambda A, b: torch.matmul(A, b[..., None])[..., 0]
    camera_directions = mat_vec_mul(pix2cam, pixel_to_direction(x, y))
    if distortion_params != None:
        x, y = _radial_and_tangential_undistort(
            camera_directions[..., 0], camera_directions[..., 1], **distortion_params
        )

    camera_directions = pixel_to_direction(x, y)
    # Switch from COLMAP (right, down, fwd) to OpenGL (right, up, back) frame.
    camera_directions = torch.matmul(
        camera_directions, torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    )

    # Apply camera rotation matrices.
    camera_directions = mat_vec_mul(pose[:3, :3], camera_directions)

    origins = torch.broadcast_to(pose[:3, -1], camera_directions.shape)
    viewdirs = camera_directions / torch.linalg.norm(
        camera_directions, axis=-1, keepdims=True
    )
    return origins.reshape(-1, 3), viewdirs.reshape(-1, 3)


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


# https://github.com/bmild/nerf/issues/34
def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

    Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    """
    t = poses[:, :3, 3].cpu().detach().numpy()
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    # numpy works with complex numbers sorting, whereas torch does not
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(torch.Tensor([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    transform = torch.from_numpy(transform)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = torch.concatenate([transform, torch.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = torch.diag(torch.Tensor([1, -1, -1])) @ poses_recentered
        transform = torch.diag(torch.Tensor([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1.0 / torch.max(torch.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    # transform = torch.diag(torch.Tensor([scale_factor] * 3 + [1])) @ transform
    return poses_recentered


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = torch.broadcast_to(torch.Tensor([0, 0, 0, 1.0]), p[..., :1, :4].shape)
    return torch.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def _compute_residual_and_jacobian(x, y, xd, yd, k1, k2, k3, k4, p1, p2):
    """Auxiliary function of radial_and_tangential_undistort()."""
    # Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd,
    yd,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    k4: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10,
):
    """Computes undistorted (x, y) from (xd, yd)."""
    # From https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # Initialize from the distorted point.
    x = torch.clone(xd)
    y = torch.clone(yd)

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(
            torch.abs(denominator) > eps,
            x_numerator / denominator,
            torch.zeros_like(denominator),
        )
        step_y = torch.where(
            torch.abs(denominator) > eps,
            y_numerator / denominator,
            torch.zeros_like(denominator),
        )

        x = x + step_x
        y = y + step_y

    return x, y


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


def psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse == 0:
        mse = 1e-5
    return -10.0 * torch.log10(torch.tensor(mse, dtype=torch.float32))
