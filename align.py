import numpy as np
import PIL
import scipy
import torch
from torchvision.transforms import functional as F


def get_landmark(img, predictor):
    img = img * 255.0
    landmarks = predictor(img)[0]

    return landmarks


def align_face(img: torch.Tensor, predictor):
    """
    :param img: array
    :return: PIL Image
    """
    lm = get_landmark(img, predictor).detach().cpu()
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
    x = eye_to_eye - torch.flipud(eye_to_mouth) * torch.tensor([-1, 1])
    x /= torch.hypot(*x)
    x *= max(torch.hypot(*eye_to_eye) * 2.0, torch.hypot(*eye_to_mouth) * 1.8)
    y = torch.flipud(x) * torch.tensor([-1, 1])
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
        img = F.resize(img, rsize)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = torch.max((torch.round(qsize * 0.1)), torch.tensor(3))
    crop = [
        torch.floor(torch.min(quad[:, 0])),
        torch.floor(torch.min(quad[:, 1])),
        torch.ceil(torch.max(quad[:, 0])),
        torch.ceil(torch.max(quad[:, 1])),
    ]
    crop = (
        torch.max(crop[0] - border, torch.tensor(0)),
        torch.max(crop[1] - border, torch.tensor(0)),
        torch.min(crop[2] + border, torch.tensor(img.shape[0])),
        torch.min(crop[3] + border, torch.tensor(img.shape[1])),
    )
    if crop[2] - crop[0] < img.shape[0] or crop[3] - crop[1] < img.shape[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (
        int(torch.floor(torch.min(quad[:, 0]))),
        int(torch.floor(torch.min(quad[:, 1]))),
        int(torch.ceil(torch.max(quad[:, 0]))),
        int(torch.ceil(torch.max(quad[:, 1]))),
    )
    pad = torch.tensor(
        [
            torch.max(-pad[0] + border, torch.tensor(0)),
            torch.max(-pad[1] + border, torch.tensor(0)),
            torch.max(pad[2] - img.shape[0] + border, torch.tensor(0)),
            torch.max(pad[3] - img.shape[1] + border, torch.tensor(0)),
        ],
    )
    if enable_padding and torch.max(pad) > border - 4:
        pad = torch.maximum(pad, torch.round(qsize * 0.3))
        img = F.pad(
            img.permute(2, 0, 1).to(torch.float32),
            (
                int(pad[0].item()),
                int(pad[1].item()),
                int(pad[2].item()),
                int(pad[3].item()),
            ),
            padding_mode="reflect",
        )
        c, h, w = img.shape
        y = torch.arange(h).view(-1, 1, 1)  # Shape: (h, 1, 1)
        x = torch.arange(w).view(1, -1, 1)  # Shape: (1, w, 1)
        mask = (
            torch.maximum(
                1.0
                - torch.minimum(
                    x.to(torch.float32) / pad[0], (w - 1 - x).to(torch.float32) / pad[2]
                ),
                1.0
                - torch.minimum(
                    y.to(torch.float32) / pad[1], (h - 1 - y).to(torch.float32) / pad[3]
                ),
            )
            .permute(2, 0, 1)
            .to("cuda")
        )
        blur = qsize * 0.02
        kernel = int(1.0 * blur + 0.5)

        img += (
            F.gaussian_blur(
                img, kernel_size=(kernel + (1 - kernel % 2)), sigma=[blur, blur]
            )
            - img
        ) * torch.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += torch.median(torch.flatten(img, 0, 1), 0).values * torch.clip(
            mask, 0.0, 1.0
        )
        img = torch.clip(torch.round(img), 0, 255).to(torch.uint8)
        quad += pad[:2]

    # Transform.
    img = F.resize(img, (transform_size, transform_size), antialias=True)
    """
    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    """
    if output_size < transform_size:
        img = F.resize(img, (output_size, output_size), antialias=True)

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

    left = max(torch.round(center[0] - padding[0]).item(), 0) // 8 * 8
    right = min(torch.round(center[0] + padding[1]).item(), w) // 8 * 8
    top = max(torch.round(center[1] - padding[2]).item(), 0) // 8 * 8
    bottom = min(torch.round(center[1] + padding[3]).item(), h) // 8 * 8

    return h, w, top, bottom, left, right, scale
