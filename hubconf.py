dependencies = ['torch']
from wav2clip.model.encoder import ResNetExtractor


MODEL_WEIGHTS_URL = ''


def wav2clip(pretrained=False, **kwargs):
    
    if pretrained:
        model = ResNetExtractor(checkpoint_path=MODEL_WEIGHTS_URL,
                                scenario='frozen',
                                transform=True,
                                **kwargs)
    else:
        model = ResNetExtractor(scenario='supervise', **kwargs)
    return model
