## Train command

```
python3 train.py \
--n_ray_samples 12 --downscale_factor 7 \
--batch_size 22500 --dataset_type blender \
--train_dataset_path /home/kolek/Edu/project/nerf_synthetic/lego \
--num_epochs 10 --save_validation_imgs True --show_validation_imgs False
```

## Render command

```
python3 render.py
```
