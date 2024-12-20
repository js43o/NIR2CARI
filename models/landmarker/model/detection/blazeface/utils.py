import numpy as np
from torchvision.transforms import functional as F, InterpolationMode


def image_resize(image, width: int = 0, height: int = 0):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]
    dim = (w, h)

    # if both the width and height are None, then return the
    # original image
    if width == 0 and height == 0:
        return image

    # check to see if the width is None
    if width == 0:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (height, int(w * r))

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (int(h * r), width)

    # resize the image
    # resized = cv2.resize(image, dim, interpolation=inter)
    resized = F.resize(
        image.permute(2, 0, 1),
        dim,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).permute(1, 2, 0)

    # return the resized image
    return resized


def resize_and_crop_image(image, dim: int):
    if image.shape[0] > image.shape[1]:
        img = image_resize(image, width=dim)
        yshift, xshift = (image.shape[0] - image.shape[1]) // 2, 0
        y_start = (img.shape[0] - img.shape[1]) // 2
        y_end = y_start + dim
        return img[y_start:y_end, :, :], (xshift, yshift)
    else:
        img = image_resize(image, height=dim)
        yshift, xshift = 0, (image.shape[1] - image.shape[0]) // 2
        x_start = (img.shape[1] - img.shape[0]) // 2
        x_end = x_start + dim
        return img[:, x_start:x_end, :], (xshift, yshift)


def resize_and_crop_batch(frames, dim):
    """
    Center crop + resize to (dim x dim)
    inputs:
        - frames: list of images (numpy arrays)
        - dim: output dimension size
    """
    smframes = []
    xshift, yshift = 0, 0
    for i in range(len(frames)):
        smframe, (xshift, yshift) = resize_and_crop_image(frames[i], dim)
        smframes.append(smframe)
    smframes = np.stack(smframes)
    return smframes, (xshift, yshift)
