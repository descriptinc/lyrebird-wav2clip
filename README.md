# Wav2CLIP

:construction: WIP :construction:

Official implementation of the paper WAV2CLIP: LEARNING ROBUST AUDIO REPRESENTATIONS FROM CLIP [:page_facing_up:](https://arxiv.org/abs/2110.11499) [:link:](https://descriptinc.github.io/lyrebird-wav2clip)

Ho-Hsiang Wu, Prem Seetharaman, Kundan Kumar, Juan Pablo Bello

We propose Wav2CLIP, a robust audio representation learning method by distilling from Contrastive Language-Image Pre-training (CLIP). We systematically evaluate Wav2CLIP on a variety of audio tasks including classification, retrieval, and generation, and show that Wav2CLIP can outperform several publicly available pre-trained audio representation algorithms. Wav2CLIP projects audio into a shared embedding space with images and text, which enables multimodal applications such as zero-shot classification, and cross-modal retrieval. Furthermore, Wav2CLIP needs just ~10% of the data to achieve competitive performance on downstream tasks compared with fully supervised models, and is more efficient to pre-train than competing methods as it does not require learning a visual model in concert with an auditory model. Finally, we demonstrate image generation from Wav2CLIP as qualitative assessment of the shared embedding space. Our code and model weights are open sourced and made available for further applications.

## Installation

```
pip install wav2clip
```

## Usage

### Clip-Level Embeddings
```
import wav2clip

model = wav2clip.get_model()
embeddings = wav2clip.embed_audio(audio, model)
```

### Frame-Level Embeddings
```
import wav2clip

model = wav2clip.get_model(frame_length=16000, hop_length=16000)
embeddings = wav2clip.embed_audio(audio, model)
```
