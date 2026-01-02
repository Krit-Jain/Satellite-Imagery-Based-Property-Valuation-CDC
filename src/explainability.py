import torch
import torch.nn.functional as F
import cv2
import numpy as np

def compute_gradcam(activations, gradients):
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1).squeeze()
    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()
    cam = cam / cam.max()
    return cam
