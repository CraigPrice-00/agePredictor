import torch.nn as nn
import torchvision.models as models


def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1),
        nn.Sigmoid()  # normalized age in [0, 1]
    )
    return model