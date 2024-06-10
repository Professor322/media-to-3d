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


def get_rays(H, W, pix2cam, pose):
    """
    H - height
    W - width
    pix2cam - translation matrix from pixels to camera coordinates (3, 3)
    pose - tranlsation matrix from camera coordinates to world coordinates (3, 4)
    """
    x, y = get_pixel_coordinates(H, W)

    def pixel_to_direction(x, y):
        return torch.stack([x, y, torch.ones_like(x)], dim=-1)

    mat_vec_mul = lambda A, b: torch.matmul(A, b[..., None])[..., 0]
    camera_directions = mat_vec_mul(pix2cam, pixel_to_direction(x, y))
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
