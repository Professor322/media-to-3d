import lightning as L

from nerf_system import NerfSystem


def train(config):
    if config.train_checkpoint_path != "":
        nerfsys = NerfSystem.load_from_checkpoint(config.train_checkpoint_path)
    else:
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
    return trainer.checkpoint_callback.best_model_path
