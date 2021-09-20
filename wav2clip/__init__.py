import torch

from .model.encoder import ResNetExtractor


MODEL_URL = 'https://www.dropbox.com/s/94fqdi4lgi8vu19/ResNet_CX.ckpt?dl=1'


def get_model(device='cpu', pretrained=True, frame_length=None, hop_length=None):
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(MODEL_URL,
                                                        map_location=device,
                                                        progress=True)
        model = ResNetExtractor(checkpoint=checkpoint,
                                scenario='frozen',
                                transform=True,
                                frame_length=frame_length,
                                hop_length=hop_length)
    else:
        model = ResNetExtractor(scenario='supervise',
                                frame_length=frame_length,
                                hop_length=hop_length)
    return model


def get_audio_embedding(audio, model):
    return model(audio)
