import torch


class PositionalEncoder(torch.nn.Module):
    """
    Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
    (Different from the paper, prepend input 'x' by default)
    Args:
        input_channels (int): number of input channels
        num_freqs (int): `L_d=4` for viewing direcion, `L_x=10` for 3D-coordinate
        log_scale (bool):
        First take power of 2 for 0 to 9 and then split equally (log_scale=False)
        or choose to generate 0-9 first and then take power of 2 separately
    """

    def __init__(
        self,
        input_channels,
        num_freqs,
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_channels = input_channels
        self.encode_fn = [torch.sin, torch.cos]
        self.output_channels = input_channels * (len(self.encode_fn) * num_freqs + 1)
        # mapping from R^input_channels to R^(num_freq * 2)
        # frequency interval derrived from 5.1 Positional Encoding. Eq (4)
        self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (ray_cnt, num_sample, self.in_channels)
        Outputs:
            out: (ray_cnt, num_sample, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.encode_fn:
                out += [func(freq * x)]

        return torch.cat(out, -1)
