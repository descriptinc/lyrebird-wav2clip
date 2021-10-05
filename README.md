# Wav2CLIP

:construction: WIP :construction:

Official implementation of the paper WAV2CLIP: LEARNING ROBUST AUDIO REPRESENTATIONS FROM CLIP [:page_facing_up:]() [:link:](https://descriptinc.github.io/lyrebird-wav2clip)

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
