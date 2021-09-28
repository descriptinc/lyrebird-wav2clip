import glob
import math
import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import IterableDataset


class VGGSoundAudioVisualDataset(IterableDataset):
    def __init__(
        self, split: str = "train", duration: int = 10, sample_rate: int = 16000
    ):
        super(VGGSoundAudioVisualDataset).__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.fps = 30

        if split in ("train", "valid"):
            self.split = "train"
            files = self.get_overlapping_files(self.split)
            if split == "train":
                files = files[:-500]
                random.shuffle(files)
            elif split == "valid":
                files = files[-500:]
        elif split == "test":
            self.split = split
            files = self.get_overlapping_files(self.split)
        else:
            assert False
        self.files = files

    def get_overlapping_files(self, split):
        audio_files = glob.glob(
            "{}/data/VGGSound/Wav/{}/*.wav".format(os.environ["ARTIFACT_ROOT"], split)
        )
        video_files = glob.glob(
            "{}/data/VGGSound/Embeddings/{}/video/*.npy".format(
                os.environ["ARTIFACT_ROOT"], split
            )
        )
        files = sorted(
            list(
                set([f.split("/")[-1].split(".")[0] for f in audio_files])
                & set([f.split("/")[-1].split(".")[0] for f in video_files])
            )
        )
        return files

    def __iter__(self):
        for f in self.files:
            audio, _ = librosa.load(
                "{}/data/VGGSound/Wav/{}/{}.wav".format(
                    os.environ["ARTIFACT_ROOT"], self.split, f
                ),
                sr=self.sample_rate,
            )
            video = np.load(
                "{}/data/VGGSound/Embeddings/{}/video/{}.npy".format(
                    os.environ["ARTIFACT_ROOT"], self.split, f
                )
            )
            num_audio_samples = self.duration * self.sample_rate
            num_video_samples = self.duration * self.fps
            if self.duration < 10:
                if (
                    audio.shape[0] >= num_audio_samples
                    and video.shape[0] >= num_video_samples
                ):
                    audio_index = random.randint(0, audio.shape[0] - num_audio_samples)
                    video_index = int(
                        np.floor((audio_index / self.sample_rate) * self.fps)
                    )
                    audio_slice = slice(audio_index, audio_index + num_audio_samples)
                    video_slice = slice(video_index, video_index + num_video_samples)
                    if (
                        audio[audio_slice].shape[0] == num_audio_samples
                        and video[video_slice, :].shape[0] == num_video_samples
                    ):
                        yield audio[audio_slice], video[video_slice, :]
            elif self.duration == 10:
                if (
                    audio.shape[0] == num_audio_samples
                    and video.shape[0] == num_video_samples
                ):
                    yield audio, video
            else:
                assert False


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    files = dataset.files
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((len(files)) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.files = files[
        worker_id * per_worker : min(worker_id * per_worker + per_worker, len(files))
    ]
