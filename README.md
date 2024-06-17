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
--train_dataset_path '/home/kolek/Edu/project/nerf_crocs2' \
--num_epochs 10 --save_validation_imgs True --show_validation_imgs False
```
## Render command

```
python3 render.py --render_checkpoint_path ./lightning_logs/version_23/checkpoints/epoch=9-step=580.ckpt \
--render_type video --fps 30 --video_duration 5 --output_path .
```

# Telegram bot
```
BOT_TOKEN=<TOKEN> python3 bot.py
```
