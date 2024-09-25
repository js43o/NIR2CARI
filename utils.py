import numpy as np
import torch
from torchvision.transforms import functional as F
from PIL import Image
import cv2
from typing import Union


def yiq_from_image(img: torch.Tensor):
    B = img[0, ...]
    G = img[1, ...]
    R = img[2, ...]
    Y, I, Q = rgb_to_yiq(R, G, B)

    return Y, I, Q


def rgb_to_yiq(R, G, B):
    Y = 0.229 * R + 0.587 * G + 0.114 * B
    I = 0.595716 * R - 0.274453 * G - 0.321263 * B
    Q = 0.221456 * R - 0.522591 * G + 0.311135 * B

    return Y, I, Q


def yiq_to_rgb(Y, I, Q):
    R = Y + 0.9563 * I + 0.621 * Q
    G = Y - 0.2721 * I - 0.6474 * Q
    B = Y - 1.1070 * I + 1.7046 * Q

    return R, G, B


def extend_to_three_channel(channel: torch.Tensor):
    return channel.unsqueeze(0).tile([3, 1, 1])


def resize_and_pad(img, size: int):
    c, h, w = img.squeeze().shape

    if h > w:
        w = int(w * (size / h))
        h = size
    else:
        h = int(h * (size / w))
        w = size

    py = (size - h) // 2
    px = (size - w) // 2

    img = F.resize(img, (h, w), antialias=True)
    img = F.pad(
        img,
        [px, py, px + (size - (w + px * 2)), py + (size - (h + py * 2))],
        padding_mode="symmetric",
    )

    return img


def get_keys(d, name):
    if "state_dict" in d:
        d = d["state_dict"]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name)] == name}
    return d_filt


def cv2_to_tensor(input: cv2.typing.MatLike):
    image = np.array(input)
    image[...] = image[..., ::-1]  # BGR to RGB
    image = pil_to_tensor(image)

    return image


def tensor_to_cv2(input: torch.Tensor):
    if input.device != "cpu":
        input = input.detach().cpu()
    image = tensor_to_pil(input)
    image = np.array(image)
    image[...] = np.array(image)[..., ::-1]

    return image


def pil_to_tensor(input: Union[Image.Image, np.ndarray]):
    image = torch.tensor(input, dtype=torch.float32)
    image = image.permute(2, 0, 1) / 255.0

    return image


def tensor_to_pil(input: torch.Tensor):
    image = input.permute(1, 2, 0) * 255.0
    image = np.array(image, dtype=np.uint8)
    image = Image.fromarray(image)

    return image
