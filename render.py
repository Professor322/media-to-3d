import argparse
import sys

# this is ugly but it works
sys.path.insert(1, "./src")
from renderer import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render_checkpoint_path",
        type=str,
        default="",
        help="Checkpoint used to load the model",
    )
    parser.add_argument(
        "--render_type",
        type=str,
        default="video",
        choices={"video", "mesh"},
        help="Type of render to do",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per seconds if video is chosen as render type",
    )
    parser.add_argument(
        "--video_duration", type=int, default=5, help="How long the video to render"
    )
    parser.add_argument(
        "--output_path", type=str, default="", help="Path to save resulting render"
    )
    config = parser.parse_args()
    render(config)


if __name__ == "__main__":
    main()
