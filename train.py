import argparse
import sys

# this is ugly but it works
sys.path.insert(1, "./src")
from trainer import train


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_size_ray",
        type=int,
        help="size of the ray for the model, will not be used when position encoding is set to True",
        default=3,
    )
    parser.add_argument(
        "--input_size_direction",
        type=int,
        help="size of the direction for the model, will not be used when position encoding is set to True",
        default=3,
    )
    parser.add_argument(
        "--n_ray_samples",
        type=int,
        help="How many samples for each ray to create",
        default=12,
    )
    parser.add_argument(
        "--downscale_factor",
        type=float,
        help="how much original dataset requires to be resized",
        default=1,
    )
    parser.add_argument(
        "--use_hierarchical_sampling",
        type=str2bool,
        help="use hierarchical sampling (not supported)",
        default=False,
    )
    parser.add_argument(
        "--batch_size", type=int, help="batch size for training", default=1
    )
    parser.add_argument(
        "--use_positional_encoding",
        type=str2bool,
        help="use positional encoding",
        default=True,
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices={"real", "blender"},
        help="Type of dataset that will be in use: blender for synthetic data, real for real data",
        default="blender",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        help="Where to find dataset for training",
        default="",
    )
    parser.add_argument(
        "--num_epochs", type=int, help="Number of epochs to run", default=1
    )
    parser.add_argument(
        "--save_validation_imgs",
        type=str2bool,
        help="Saves validation imgs",
        default=True,
    )
    parser.add_argument(
        "--show_validation_imgs",
        type=str2bool,
        help="Show validation imgs on validation epoch. Not recommend to use in the command line",
        default=False,
    )
    parser.add_argument(
        "--train_checkpoint_path",
        type=str,
        default="",
        help="Checkpoint to start training from",
    )

    config = parser.parse_args()
    train(config)


if __name__ == "__main__":
    main()
