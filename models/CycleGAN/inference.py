import os
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

input_shape = (3, 1024, 1024)
n_residual_blocks = 9

input_path = "output/vtoonify"
output_path = "output/cyclegan"


def sample_images(path, filename, generator):
    generator.eval()
    image = np.float32(cv2.imread(path))
    Y, I, Q = yiq_from_image(image)

    print(Y.shape, I.shape, Q.shape)

    cv2.imwrite("%s/%s_Y.png" % (output_path, filename), np.array(Y))

    luminance = extend_triple(Y) / 255.0
    real = transform(luminance).type(Tensor).unsqueeze(dim=0)
    generated = generator(real)

    print(generated.shape)

    generated = generated.detach().cpu().squeeze().permute(1, 2, 0).numpy() * 120 + 120
    cv2.imwrite("%s/%s_luminance.png" % (output_path, filename), generated)
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)

    R, G, B = yiq_to_rgb(generated, I, Q)
    synthesized = np.array([B, G, R]).transpose(1, 2, 0)
    cv2.imwrite("%s/%s.png" % (output_path, filename), synthesized)


def main():
    os.makedirs("%s", exist_ok=True)
    generator = GeneratorResNet(input_shape, n_residual_blocks)
    if cuda:
        generator = generator.cuda()

    generator.load_state_dict(torch.load("models/CycleGAN/checkpoint/generator.pth"))

    for filename in os.listdir(os.path.join(input_path)):
        sample_images(os.path.join(input_path, filename), filename, generator)
