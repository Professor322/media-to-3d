import os
import shutil
import subprocess

import cv2
import torch

import background_remover


class VideoPreprocessor:
    """
    This class implements logic that extracts images of the object from a video
    It also runs COLMAP to estimate camera positions
    """

    def __init__(
        self, video_path, images_path, colmap_script_path, num_frames, remove_background
    ):
        self.video_path = video_path
        self.images_path = images_path
        self.colmap_script_path = colmap_script_path
        self.remove_background = remove_background
        assert os.path.exists(self.video_path)
        self.video_capture = cv2.VideoCapture(self.video_path)

        self.total_num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # num images to extract
        self.num_frames = num_frames
        if self.total_num_frames < self.num_frames:
            self.video_capture.release()
            raise Exception(
                "Requested number of frames greater than total number of frames"
            )
        self.frame_indicies = torch.linspace(
            0, self.total_num_frames - 1, self.num_frames, dtype=torch.int32
        )

        # check for existence of image_path dir
        if os.path.exists(self.images_path):
            # delete everything in this folder
            print(f"{self.images_path} exists, deleting")
            shutil.rmtree(self.images_path)

        os.mkdir(self.images_path)
        os.mkdir(self.images_path + "/images/")

        current_frame_index = 0
        while self.video_capture.isOpened() and current_frame_index < len(
            self.frame_indicies
        ):
            self.video_capture.set(
                cv2.CAP_PROP_POS_FRAMES, self.frame_indicies[current_frame_index].item()
            )
            success, image = self.video_capture.read()

            if success:
                ret = cv2.imwrite(
                    f"{self.images_path}/images/frame_{self.frame_indicies[current_frame_index]}.jpg",
                    image,
                )
                if ret == False:
                    self.video_capture.release()
                    raise Exception(f"Failure to write to {self.images_path}")
            else:
                self.video_capture.release()
                raise Exception("failed to read a frame")
            current_frame_index += 1

        self.video_capture.release()
        print(f"Created {current_frame_index} frames")

        print("Running COLMAP to estimate camera positions...")
        subprocess.run(
            f"bash {self.colmap_script_path} {self.images_path}",
            shell=True,
            executable="/bin/bash",
        )
        if self.remove_background:
            print("Removing background...")
            os.mkdir(self.images_path + "/images_no_background/")
            for img_frame_index in self.frame_indicies:
                background_remover.remove_bg(
                    src_img_path=f"{self.images_path}/images/frame_{img_frame_index}.jpg",
                    out_img_path=f"{self.images_path}/images_no_background/frame_{img_frame_index}.jpg",
                )
            os.rename(f"{self.images_path}/images", f"{self.images_path}/original_images")
            os.rename(
                f"{self.images_path}/images_no_background", f"{self.images_path}/images"
            )

    def delete_processed_images(self):
        shutil.rmtree(self.images_path)
