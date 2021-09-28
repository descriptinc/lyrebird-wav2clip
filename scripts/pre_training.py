import os
import time

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from wav2clip.pre_training.dataset import (
    VGGSoundAudioVisualDataset,
)
from wav2clip.pre_training.dataset import worker_init_fn
from wav2clip.pre_training.model import ResNetDistillation


if __name__ == "__main__":
    import argbind

    args = argbind.load_args("conf/distillation_args.yaml")

    ResNetDistillation = argbind.bind(ResNetDistillation)
    func_args = argbind.parse_args()
    args.update(func_args)
    seed_everything(args["random_state"])

    exp_name = "-".join(
        ["{}:{}".format(k.split(".")[-1], v) for k, v in args.items()]
        + [str(int(time.time()))]
    )
    project_root = "{}/models/{}".format(os.environ["ARTIFACT_ROOT"], exp_name)
    os.makedirs(project_root, exist_ok=True)

    tensorboard_logger = TensorBoardLogger(save_dir="{}/logs/".format(project_root))

    dirpath = "{}/models/".format(project_root)
    filename = "{epoch}-{val_loss:.4f}"

    trainer = Trainer(
        logger=tensorboard_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args["early_stopping_patience"]),
            ModelCheckpoint(
                dirpath=dirpath, filename=filename, monitor="val_loss", save_top_k=-1
            ),
        ],
        gpus=args["num_gpus"],
        accelerator="dp",
        max_epochs=100,
    )
    train_loader = DataLoader(
        VGGSoundAudioVisualDataset(split="train"),
        num_workers=args["num_workers"],
        batch_size=args["batch_size"],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    valid_loader = DataLoader(
        VGGSoundAudioVisualDataset(split="valid"),
        num_workers=args["num_workers"],
        batch_size=args["batch_size"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    with argbind.scope(args):
        distillation = ResNetDistillation(args)
    trainer.fit(distillation, train_loader, valid_loader)
