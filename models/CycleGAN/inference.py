import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from .models import *
from .luminance import *

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def sample_images(image, generator):
    Y, I, Q = yiq_from_image(image)

    luminance = extend_to_three_channel(Y) / 255.0
    real = transform(luminance).type(Tensor).unsqueeze(dim=0)
    generated = generator(real)

    generated = generated.detach().cpu().squeeze().permute(1, 2, 0).numpy() * 120 + 120
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)

    R, G, B = yiq_to_rgb(generated, I, Q)
    synthesized = np.array([B, G, R]).transpose(1, 2, 0).clip(0, 255).astype(np.uint8)

    return synthesized
