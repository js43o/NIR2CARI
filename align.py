import numpy as np
import PIL
import scipy
import torch
from torchvision.transforms import functional as F


def get_landmark(img, predictor):
    img = img.permute(1, 2, 0) * 255.0
    landmarks = predictor(img)[0]

    return landmarks


def align_face(img: torch.Tensor, predictor):
    """
    :param img: array
    :return: PIL Image
    """
    lm = get_landmark(img, predictor)
    if lm is None:
        return lm

    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = torch.mean(lm_eye_left, dim=0)
    eye_right = torch.mean(lm_eye_right, dim=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - torch.flipud(eye_to_mouth) * [-1, 1]
    x /= torch.hypot(*x)
    x *= max(torch.hypot(*eye_to_eye) * 2.0, torch.hypot(*eye_to_mouth) * 1.8)
    y = torch.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = torch.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = torch.hypot(*x) * 2

    output_size = 256
    transform_size = 256
    enable_padding = True

    # Shrink.
    shrink = int(torch.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(torch.round(float(img.size[0]) / shrink)),
            int(torch.round(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(torch.round(qsize * 0.1)), 3)
    crop = (
        int(torch.floor(min(quad[:, 0]))),
        int(torch.floor(min(quad[:, 1]))),
        int(torch.ceil(max(quad[:, 0]))),
        int(torch.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (
        int(torch.floor(min(quad[:, 0]))),
        int(torch.floor(min(quad[:, 1]))),
        int(torch.ceil(max(quad[:, 0]))),
        int(torch.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = torch.maximum(pad, int(torch.round(qsize * 0.3)))
        img = F.pad(
            img.to(torch.float32),
            (pad[0], pad[1], pad[2], pad[3]),
            padding_mode="reflect",
        )
        h, w, _ = img.shape
        y = torch.arange(h).view(-1, 1, 1)  # Shape: (h, 1, 1)
        x = torch.arange(w).view(1, -1, 1)  # Shape: (1, w, 1)
        mask = torch.maximum(
            1.0
            - torch.minimum(
                x.to(torch.float32) / pad[0], (w - 1 - x).to(torch.float32) / pad[2]
            ),
            1.0
            - torch.minimum(
                y.to(torch.float32) / pad[1], (h - 1 - y).to(torch.float32) / pad[3]
            ),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * torch.clip(
            mask * 3.0 + 1.0, 0.0, 1.0
        )
        img += (torch.median(img, axis=(0, 1)) - img) * torch.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(
            (torch.clip(torch.round(img), 0, 255)).to(torch.uint8), "RGB"
        )
        quad += pad[:2]

    # Transform.
    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    return img


def get_video_crop_parameter(img, predictor, padding=[200, 200, 200, 200]):
    lm = get_landmark(img, predictor)
    if lm is None:
        return None

    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise

    scale = 64.0 / (torch.mean(lm_eye_right[:, 0]) - torch.mean(lm_eye_left[:, 0]))
    center = (
        (torch.mean(lm_eye_right, dim=0) + torch.mean(lm_eye_left, dim=0)) / 2
    ) * scale
    h = int(torch.round(img.shape[0] * scale).item())
    w = int(torch.round(img.shape[1] * scale).item())

    left = max(torch.round(center[0] - padding[0]).item(), 0)
    left = torch.div(left, 64, rounding_mode="floor")

    right = min(torch.round(center[0] + padding[1]).item(), w)
    right = torch.div(right, 64, rounding_mode="floor")

    top = max(torch.round(center[1] - padding[2]).item(), 0)
    top = torch.div(top, 64, rounding_mode="floor")

    bottom = min(torch.round(center[1] + padding[3]).item(), h)
    bottom = torch.div(bottom, 64, rounding_mode="floor")

    return h, w, top, bottom, left, right, scale
