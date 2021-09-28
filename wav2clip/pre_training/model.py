import pytorch_lightning as pl
import torch

from ..model.encoder import MLPLayers
from ..model.resnet import BasicBlock
from ..model.resnet import ResNet
from .loss import CLIPLoss1D


class LightningBase(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        return {
            "avg_val_loss": avg_loss,
            "log": {"val_loss": avg_loss},
            "progress_bar": {"val_loss": avg_loss},
        }

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss", avg_loss, prog_bar=True)
        return {
            "avg_test_loss": avg_loss,
            "log": {"test_loss": avg_loss},
            "progress_bar": {"test_loss": avg_loss},
        }

    def configure_optimizers(self):
        if self.args["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.args.learning_rate, momentum=0.9
            )
        elif self.args["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.args["learning_rate"]
            )
        else:
            assert False
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.args["lr_scheduler_patience"],
            min_lr=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class ResNetDistillation(LightningBase):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        assert args["loss"] in (
            "clip_loss",
            "clip_loss_x",
            "clip_loss_i",
        )
        self.loss = args["loss"]
        self.args = args

        self.audio_encoder = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            num_classes=309,
            pool="avgpool",
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
        )
        self.image_transform = None
        self.audio_transform = None

        if self.loss == "clip_loss":
            self.loss_fn = CLIPLoss1D()
        elif self.loss == "clip_loss_i":
            self.image_transform = MLPLayers(
                units=args["MLPLayers.units"], dropout=args["MLPLayers.dropout"]
            )
            self.loss_fn = CLIPLoss1D()
        elif self.loss == "clip_loss_x":
            self.image_transform = MLPLayers(
                units=args["MLPLayers.units"], dropout=args["MLPLayers.dropout"]
            )
            self.audio_transform = MLPLayers(
                units=args["MLPLayers.units"], dropout=args["MLPLayers.dropout"]
            )
            self.loss_fn_i = CLIPLoss1D()
            self.loss_fn_a = CLIPLoss1D()
        else:
            assert False

    def forward(self, audio, images):
        audio_output = self.audio_encoder(audio.float())
        image_output = torch.mean(images.float(), 1)
        if self.loss == "clip_loss_i":
            image_output = self.image_transform(image_output)
        elif self.loss == "clip_loss_x":
            transformed_image = self.image_transform(image_output)
            transformed_audio = self.audio_transform(audio_output)
            return audio_output, image_output, transformed_audio, transformed_image
        return audio_output, image_output

    def step(self, batch, batch_idx):
        audio, images = batch
        if self.loss == "clip_loss_x":
            audio_out, image_out, transformed_audio, transformed_image = self.forward(
                audio, images
            )
            loss = (
                self.loss_fn_a(audio_out, transformed_image)
                + self.loss_fn_i(transformed_audio, image_out)
            ) / 2
        else:
            audio_out, image_out = self.forward(audio, images)
            loss = self.loss_fn(audio_out, image_out)
        return loss
