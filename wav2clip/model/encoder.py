import librosa
import numpy as np
import torch
from torch import nn

from .resnet import BasicBlock
from .resnet import ResNet


class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


class ResNetExtractor(nn.Module):
    def __init__(
        self,
        checkpoint: dict = None,
        scenario: str = "frozen",
        transform: bool = False,
        mlp_layers_units: list = [512, 512, 512],
        mlp_layers_dropout: float = 0.1,
        frame_length: int = None,
        hop_length: int = None,
    ):
        super().__init__()
        assert scenario in ("supervise", "frozen", "finetune")
        self.scenario = scenario
        self.frame_length = frame_length
        self.hop_length = hop_length

        self.encoder = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            pool="avgpool",
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
        )
        self.transform = None

        if checkpoint and self.scenario != "supervise":
            self.encoder.load_state_dict(
                {
                    ".".join(k.split(".")[1:]): v
                    for k, v in checkpoint.items()
                    if k.startswith("audio_encoder")
                }
            )
            if self.scenario == "frozen":
                self.encoder.eval()
            else:
                self.encoder.train()

            if transform:
                self.transform = MLPLayers(
                    units=mlp_layers_units,
                    dropout=mlp_layers_dropout,
                )
                self.transform.load_state_dict(
                    {
                        ".".join(k.split(".")[1:]): v
                        for k, v in checkpoint.items()
                        if k.startswith("audio_transform")
                    }
                )
                if self.scenario == "frozen":
                    self.transform.eval()
                else:
                    self.transform.train()

    def forward(self, x):
        if self.frame_length and self.hop_length:
            if x.shape[0] == 1:
                frames = np.expand_dims(
                    librosa.util.frame(
                        x.cpu().numpy()[0], self.frame_length, self.hop_length
                    ),
                    axis=0,
                )
            else:
                frames = librosa.util.frame(
                    x.cpu().numpy(), self.frame_length, self.hop_length
                )
            batch_size, frame_size, frame_num = frames.shape
            feature = self.encoder(
                torch.swapaxes(torch.from_numpy(frames), 1, 2)
                .reshape(batch_size * frame_num, frame_size)
                .to(next(self.encoder.parameters()).device)
            )
            _, embedding_size = feature.shape
            if self.transform:
                feature = self.transform(feature)
            feature = torch.swapaxes(
                feature.reshape(batch_size, frame_num, embedding_size), 1, 2
            )
            if self.scenario == "frozen":
                feature.detach()
            return feature
        else:  # utterance
            feature = self.encoder(x)
            if self.transform:
                feature = self.transform(feature)
            if self.scenario == "frozen":
                feature.detach()
        return feature
