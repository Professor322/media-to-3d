import torch.nn as nn


class VolumeRenderer(nn.Module):
    """
    Module to render 2D view of our 3D model
    from the particular angle
    """

    def __init__():
        pass

    def forward(self, nerf_model, ray):
        """
        Forward path of a renderer takes output of NeRF MLP and a ray along which
        we need to create a view.
        Output of the NeRF model is a colors along a ray and a sigma (color density)
        \gamma = t_{i + 1} - t_{i} - distance between adjacent samples
        \alpha = 1 - exp(-\sigma_{i} * \gamma_{i}) (alpha compositing value)
        C_{i}(r_{i}) = \sum(T_{i}(\alpha_{i}) * c_{i})  sum from t_near to t_far \\ equation (3)
        T_{i}(\alpha) = exp(-sum(\alpha))  sum from t_near to i
        """
        pass
