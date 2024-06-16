import argparse
import sys

import lightning as L

# this is ugly but it works
sys.path.insert(1, "./src")
from nerf_system import NerfSystem


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train(config):
    nerfsys = NerfSystem(
        input_size_ray=config.input_size_ray,
        input_size_direction=config.input_size_direction,
        n_ray_samples=config.n_ray_samples,
        downscale_factor=config.downscale_factor,
        batch_size=config.batch_size,
        use_positional_encoding=config.use_positional_encoding,
        use_hierarchical_sampling=config.use_hierarchical_sampling,
        dataset_type=config.dataset_type,
        train_dataset_path=config.train_dataset_path,
        show_validation_imgs=config.show_validation_imgs,
        save_validation_imgs=config.save_validation_imgs,
    )

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        enable_model_summary=True,
        enable_progress_bar=True,
        benchmark=False,
        profiler=None,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )

    trainer.fit(nerfsys)


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
    config = parser.parse_args()
    train(config)


if __name__ == "__main__":
    main()
