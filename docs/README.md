# WAV2CLIP

Ho-Hsiang Wu, Prem Seetharaman, Kundan Kumar, Juan Pablo Bello
<div><a href="https://arxiv.org/abs/2110.11499"><img src="https://github.githubassets.com/images/icons/emoji/unicode/1f4c4.png" alt=":page_facing_up:" style="width: 32px;"></a><a href="https://github.com/descriptinc/lyrebird-wav2clip"><img src="https://github.githubassets.com/images/icons/emoji/octocat.png" alt=":octocat:" style="width: 32px;"></a></div>

## Abstract

We propose Wav2CLIP, a robust audio representation learning method by distilling from Contrastive Language-Image Pre-training (CLIP). We systematically evaluate Wav2CLIP on a variety of audio tasks including classification, retrieval, and generation, and show that Wav2CLIP can outperform several publicly available pre-trained audio representation algorithms. Wav2CLIP projects audio into a shared embedding space with images and text, which enables multimodal applications such as zero-shot classification, and cross-modal retrieval. Furthermore, Wav2CLIP needs just ~10% of the data to achieve competitive performance on downstream tasks compared with fully supervised models, and is more efficient to pre-train than competing methods as it does not require learning a visual model in concert with an auditory model. Finally, we demonstrate image generation from Wav2CLIP as qualitative assessment of the shared embedding space. Our code and model weights are open sourced and made available for further applications.

## VQGAN-CLIP Generate Samples

### ESC-50

<div><span style="width: 200px; text-align: center; display: inline-block;">Frog</span><span style="width: 200px; text-align: center; display: inline-block;">Frog</span><span style="width: 200px; text-align: center; display: inline-block;">Frog</span><span style="width: 200px; text-align: center; display: inline-block;">Frog</span></div>
<div><img src="artifacts/esc50/1-18755-A-4-frog.png" alt="1-18755-A-4" width="200"/><img src="artifacts/esc50/1-15689-B-4-frog.png" alt="1-15689-B-4" width="200"/><img src="artifacts/esc50/2-32515-B-4-frog.png" alt="2-32515-B-4" width="200"/><img src="artifacts/esc50/1-31836-B-4-frog.png" alt="1-31836-B-4" width="200"/></div>
<div><audio controls style="width: 200px;" src="artifacts/esc50/1-18755-A-4.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/1-15689-B-4.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/2-32515-B-4.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/1-31836-B-4.wav"></audio></div>

<div><span style="width: 200px; text-align: center; display: inline-block;">Church_bells</span><span style="width: 200px; text-align: center; display: inline-block;">Church_bells</span><span style="width: 200px; text-align: center; display: inline-block;">Church_bells</span><span style="width: 200px; text-align: center; display: inline-block;">Church_bells</span></div>
<div><img src="artifacts/esc50/5-219044-A-46-church_bells.png" alt="5-219044-A-46" width="200"/><img src="artifacts/esc50/1-13572-A-46-church_bells.png" alt="1-13572-A-46" width="200"/><img src="artifacts/esc50/3-139109-A-46-church_bells.png" alt="3-139109-A-46" width="200"/><img src="artifacts/esc50/2-77346-A-46-church_bells.png" alt="2-77346-A-46" width="200"/></div>
<div><audio controls style="width: 200px;" src="artifacts/esc50/5-219044-A-46.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/1-13572-A-46.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/3-139109-A-46.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/2-77346-A-46.wav"></audio></div>

<div><span style="width: 200px; text-align: center; display: inline-block;">Fireworks</span><span style="width: 200px; text-align: center; display: inline-block;">Fireworks</span><span style="width: 200px; text-align: center; display: inline-block;">Fireworks</span><span style="width: 200px; text-align: center; display: inline-block;">Fireworks</span></div>
<div><img src="artifacts/esc50/1-115545-B-48-fireworks.png" alt="1-115545-B-48" width="200"/><img src="artifacts/esc50/5-160614-C-48-fireworks.png" alt="5-160614-C-48" width="200"/><img src="artifacts/esc50/1-115545-A-48-fireworks.png" alt="1-115545-A-48" width="200"/><img src="artifacts/esc50/1-115546-A-48-fireworks.png" alt="1-115546-A-48" width="200"/></div>
<div><audio controls style="width: 200px;" src="artifacts/esc50/1-115545-B-48.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/5-160614-C-48.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/1-115545-A-48.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/1-115546-A-48.wav"></audio></div>

<div><span style="width: 200px; text-align: center; display: inline-block;">Chirping_birds</span><span style="width: 200px; text-align: center; display: inline-block;">Crow</span><span style="width: 200px; text-align: center; display: inline-block;">Wind</span><span style="width: 200px; text-align: center; display: inline-block;">Clock_alarm</span></div>
<div><img src="artifacts/esc50/1-34495-A-14-chirping_birds.png" alt="1-34495-A-14" width="200"/><img src="artifacts/esc50/1-39835-B-9-crow.png" alt="1-39835-B-9" width="200"/><img src="artifacts/esc50/2-109374-A-16-wind.png" alt="2-109374-A-16" width="200"/><img src="artifacts/esc50/1-96890-A-37-clock_alarm.png" alt="1-96890-A-37" width="200"/></div>
<div><audio controls style="width: 200px;" src="artifacts/esc50/1-34495-A-14.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/1-39835-B-9.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/2-109374-A-16.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/1-96890-A-37.wav"></audio></div>

<div><span style="width: 200px; text-align: center; display: inline-block;">Crickets</span><span style="width: 200px; text-align: center; display: inline-block;">Sheep</span><span style="width: 200px; text-align: center; display: inline-block;">Insects</span><span style="width: 200px; text-align: center; display: inline-block;">Airplane</span></div>
<div><img src="artifacts/esc50/2-96033-A-13-crickets.png" alt="2-96033-A-13" width="200"/><img src="artifacts/esc50/1-49409-A-8-sheep.png" alt="1-49409-A-8" width="200"/><img src="artifacts/esc50/1-73585-A-7-insects.png" alt="1-73585-A-7" width="200"/><img src="artifacts/esc50/2-74361-A-47-airplane.png" alt="2-74361-A-47" width="200"/></div>
<div><audio controls style="width: 200px;" src="artifacts/esc50/2-96033-A-13.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/1-49409-A-8.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/1-73585-A-7.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/2-74361-A-47.wav"></audio></div>

<div><span style="width: 200px; text-align: center; display: inline-block;">Siren</span><span style="width: 200px; text-align: center; display: inline-block;">Car_horn</span><span style="width: 200px; text-align: center; display: inline-block;">Dog</span><span style="width: 200px; text-align: center; display: inline-block;">Crying_baby</span></div>
<div><img src="artifacts/esc50/2-43806-A-42-siren.png" alt="2-43806-A-42" width="200"/><img src="artifacts/esc50/2-54086-A-43-car_horn.png" alt="2-54086-A-43" width="200"/><img src="artifacts/esc50/2-116400-A-0-dog.png" alt="2-116400-A-0" width="200"/><img src="artifacts/esc50/1-22694-A-20-crying_baby.png" alt="1-22694-A-20" width="200"/></div>
<div><audio controls style="width: 200px;" src="artifacts/esc50/2-43806-A-42.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/2-54086-A-43.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/2-116400-A-0.wav"></audio><audio controls style="width: 200px;" src="artifacts/esc50/1-22694-A-20.wav"></audio></div>

## Replicate

We also provide more examples through [Replicate](https://replicate.com/hohsiangwu/wav2clip).

## UMAP Visualizations of Various Audio Classification Tasks

<div><img src="artifacts/umap/umap_urbansound8k.png" alt="UMAP UrbanSound8K" width="400" style="background-color:#FFFFFF"/><img src="artifacts/umap/umap_esc50.png" alt="UMAP ESC-50" width="400" style="background-color:#FFFFFF"/></div>
<div><img src="artifacts/umap/umap_vggsound.png" alt="UMAP VGGSound" width="400" style="background-color:#FFFFFF"/><img src="artifacts/umap/umap_tau.png" alt="UMAP TAU" width="400" style="background-color:#FFFFFF"/></div>
