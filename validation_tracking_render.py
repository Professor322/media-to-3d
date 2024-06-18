import argparse
import os
import sys

import cv2


def render_validation(val_imgs_path, output_video_path, stop_at=0, fps=30):
    print("Rendering...")
    # read first img to capture its resolution
    img_index = 0
    video_file_name = "/validation_imgs_render.mp4"
    img = cv2.imread(f"{val_imgs_path}/img_{img_index}.png")
    H, W, channels = img.shape
    writer = cv2.VideoWriter(
        output_video_path + video_file_name,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (W, H),
    )
    writer.write(img)
    img_index += 1
    while os.path.exists(f"{val_imgs_path}/img_{img_index}.png") and img_index < stop_at:
        img = cv2.imread(f"{val_imgs_path}/img_{img_index}.png")
        writer.write(img)
        img_index += 1

    writer.release()
    print(f"Done. Validation video is saved at {output_video_path + video_file_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_imgs_path",
        type=str,
        default="",
        help="Path to the dir with validation images",
    )
    parser.add_argument(
        "--output_video_path",
        type=str,
        default="",
        help="Path where to store resulting video",
    )
    parser.add_argument(
        "--stop_at", type=int, default=sys.maxsize, help="Which frame to stop at"
    )
    config = parser.parse_args()
    render_validation(
        val_imgs_path=config.validation_imgs_path,
        output_video_path=config.output_video_path,
        stop_at=config.stop_at,
    )


if __name__ == "__main__":
    main()
