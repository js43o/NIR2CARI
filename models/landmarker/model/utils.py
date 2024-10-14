import os
import sys
import errno
import torch
from typing import List, Optional
from torchvision.transforms import functional as F, InterpolationMode

from urllib.parse import urlparse
from torch.hub import download_url_to_file, HASH_REGEX

try:
    from torch.hub import get_dir
except BaseException:
    from torch.hub import _get_torch_home as get_dir


def transform(point, center, scale: float, resolution: float, invert: bool = False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()


def crop(image, center, scale: float, resolution: float = 256.0):
    """Center crops an image or set of heatmaps

    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face

    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})

    Returns:
        [type] -- [description]
    """  # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform(torch.tensor([1.0, 1.0]), center, scale, resolution, True)
    br = transform(
        torch.tensor([resolution, resolution]), center, scale, resolution, True
    )
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
    if image.ndim > 2:
        newDim = [
            int((br[1] - ul[1]).item()),
            int((br[0] - ul[0]).item()),
            image.shape[2],
        ]
        newImg = torch.zeros(newDim, dtype=torch.float32)
    else:
        newDim = [int((br[1] - ul[1]).item()), int((br[0] - ul[0]).item())]
        newImg = torch.zeros(newDim, dtype=torch.float32)
    ht = image.shape[0]
    wd = image.shape[1]

    newX = torch.tensor(
        [max(1, -ul[0] + 1), int((min(br[0], wd) - ul[0]).item())], dtype=torch.int32
    )
    newY = torch.tensor(
        [max(1, -ul[1] + 1), int((min(br[1], ht) - ul[1]).item())], dtype=torch.int32
    )
    oldX = torch.tensor([max(1, ul[0] + 1), min(br[0], wd)], dtype=torch.int32)
    oldY = torch.tensor([max(1, ul[1] + 1), min(br[1], ht)], dtype=torch.int32)
    newImg[newY[0] - 1 : newY[1], newX[0] - 1 : newX[1]] = image[
        oldY[0] - 1 : oldY[1], oldX[0] - 1 : oldX[1], :
    ]
    newImg = F.resize(
        newImg.permute(2, 0, 1),
        size=(int(resolution), int(resolution)),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).permute(1, 2, 0)
    return newImg


# @jit(nopython=True)
def transform_np(point, center, scale: float, resolution: int, invert: bool = False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {numpy.array} -- the input 2D point
        center {numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.linalg.pinv(t).contiguous()

    new_point = torch.matmul(t, _pt)[0:2]

    return new_point.to(torch.int32)


def get_preds_fromhm(hm, center: Optional[torch.Tensor], scale: Optional[float]):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    hm_reshape = hm.reshape(B, C, H * W)
    idx = torch.argmax(hm_reshape, dim=-1).unsqueeze(-1)
    scores = torch.gather(hm_reshape, dim=-1, index=idx).squeeze(-1)
    preds, preds_orig = _get_preds_fromhm(hm, idx, center, scale)

    return preds, preds_orig, scores


# @jit(nopython=True)
def _get_preds_fromhm(hm, idx, center: Optional[torch.Tensor], scale: Optional[float]):
    """Obtain (x,y) coordinates given a set of N heatmaps and the
    coresponding locations of the maximums. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    idx += 1
    # preds = idx.repeat(2).reshape(B, C, 2).astype(np.float32)
    preds = idx.repeat_interleave(2).reshape(B, C, 2).to(torch.float32)
    preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / H) + 1

    for i in range(B):
        for j in range(C):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.tensor(
                    [
                        int((hm_[pY, pX + 1] - hm_[pY, pX - 1]).item()),
                        int((hm_[pY + 1, pX] - hm_[pY - 1, pX]).item()),
                    ]
                ).to("cuda")
                preds[i, j] += torch.sign(diff) * 0.25

    preds -= 0.5

    preds_orig = torch.zeros_like(preds)
    if center is not None and scale is not None:
        for i in range(B):
            for j in range(C):
                preds_orig[i, j] = transform_np(preds[i, j], center, scale, H, True)

    return preds, preds_orig


# Pytorch load supports only pytorch models
def load_file_from_url(
    url, model_dir=None, progress=True, check_hash=False, file_name=None
):
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file
