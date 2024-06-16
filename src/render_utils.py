import numpy as np
import torch


def trans_t(t):
    t = torch.as_tensor(t)
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]], dtype=torch.float
    )


def rot_phi(phi):
    phi = torch.as_tensor(phi)
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, torch.cos(phi), -torch.sin(phi), 0],
            [0, torch.sin(phi), torch.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float,
    )


def rot_theta(theta):
    theta = torch.as_tensor(theta)
    return torch.tensor(
        [
            [torch.cos(theta), 0, -torch.sin(theta), 0],
            [0, 1, 0, 0],
            [torch.sin(theta), 0, torch.cos(theta), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float,
    )


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).float()
        @ c2w
    )
    return c2w


def create_spherical_poses(num_poses=120):
    render_poses = []
    for th in np.linspace(0.0, 360.0, num_poses, endpoint=False):
        render_poses.append(pose_spherical(th, -30.0, 4.0))
    return render_poses
