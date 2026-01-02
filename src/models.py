import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18_feature_extractor():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()  # removes classifier
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model
