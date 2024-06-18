# Media to 3D
This is an attempt to create a service that generates 3D representation of an object from a single video or multiple images. In its core it is using Nerf model (link) <br>
Verified that it works well with synthetic data.

What is supported

* Synthetic and real datasets (colmap is used for camera positions estimations)
* Frame extractions from the video
* Validation tracking
* Checkpointing
* Telegram bot
* Docker container that wraps everything nicely

What is missing
* PDF Sampler and Fine model
* Mesh generation
* Nice rendering for real data

Note: training and validation was done locally on GTX1650TI which is not enough for this task. So, maybe with more compute you will be able to create more samples along the ray and train for much longer. This might help to get better results on the real world data

From [multinerf](https://github.com/google-research/multinerf) I took pycolmap, distorions, some render,ray utils and colmap script. <br>
From [this nerf implemenation](https://github.com/kwea123/nerf_pl) I borrowed modular design

Sample video generations from Blender dataset:<br>
![chair_render-ezgif com-video-to-gif-converter](https://github.com/Professor322/media-to-3d/assets/36162000/a5ff4648-b8c0-4012-b685-540f423893f0)
![drumset_render-ezgif com-video-to-gif-converter](https://github.com/Professor322/media-to-3d/assets/36162000/ad3be4da-1e9a-4f9d-87ce-379a22401cd1)




Sample validation tracking:
(insert videos)

## Setup


### Venv
You might need to install ffmpeg python dev tools and some jpeg tools
```
apt install ffmpeg python3.12-dev libjpeg-dev zlib1g-dev
```

Clone colmap repo
```
git clone https://github.com/rmbrualla/pycolmap.git ./src/pycolmap
```

Code was tested using `python3.12`

```
python3 -m venv .media_to_3d
source .media_to_3d/bin/activate
pip install -r requirements.txt
```

## Train command

### Blender
```
python3 train.py \
--n_ray_samples 12 --downscale_factor 7 \
--batch_size 22500 --dataset_type blender \
--train_dataset_path /home/kolek/Edu/project/nerf_synthetic/lego \
--num_epochs 10 --save_validation_imgs True --show_validation_imgs False \
--train_checkpoint_path ./lightning_logs/version_23/checkpoints/epoch=9-step=580.ckpt \
```
### Real

```
python3 train.py \
--n_ray_samples 12 --downscale_factor 6.5 \
--batch_size 21560 --dataset_type real \
--train_dataset_path <checkpoint_path> \
--num_epochs 10 --save_validation_imgs True --show_validation_imgs False
```
## Render command

```
python3 render.py --render_checkpoint_path <checkpoint_path> \
--render_type video --fps 30 --video_duration 5 --output_path .
```

## Telegram bot
```
BOT_TOKEN=<TOKEN> python3 bot.py
```
How it looks like: <br>
![telegram_bot_demo](https://github.com/Professor322/media-to-3d/assets/36162000/a291a289-fedd-48d9-9c6c-f5ff023a557f)

## Docker

### Build container

```
docker build -t media_to_3d_container .
```

### Run container
You can run newly built container through `Docker Desktop` or using
```
docker container run -e BOT_TOKEN media_to_3d_container
```
