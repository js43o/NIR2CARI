import numpy as np
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

from ...utils import load_file_from_url
from .net_blazeface import BlazeFace
from .utils import resize_and_crop_image

models_urls = {
    "blazeface_weights": "https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.pth?raw=true",
    "blazeface_anchors": "https://github.com/hollance/BlazeFace-PyTorch/blob/master/anchors.npy?raw=true",
    "blazeface_back_weights": "https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazefaceback.pth?raw=true",
    "blazeface_back_anchors": "https://github.com/hollance/BlazeFace-PyTorch/blob/master/anchorsback.npy?raw=true",
}


class BlazeFaceDetector(nn.Module):
    def __init__(
        self,
        device="cuda",
        path_to_detector=None,
        path_to_anchor=None,
        verbose=False,
        min_score_thresh=0.5,
        min_suppression_threshold=0.3,
    ):
        super().__init__()
        self.device = device
        self.verbose = verbose
        # Initialise the face detector
        if path_to_detector is None:
            model_weights = load_url(models_urls["blazeface_weights"])
            model_anchors = np.load(
                load_file_from_url(models_urls["blazeface_anchors"])
            )
        else:
            model_weights = torch.load(path_to_detector)
            model_anchors = np.load(path_to_anchor)

        self.face_detector = BlazeFace()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.load_anchors_from_npy(model_anchors, device)

        # Optionally change the thresholds:
        self.face_detector.min_score_thresh = min_score_thresh
        self.face_detector.min_suppression_threshold = min_suppression_threshold

        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, image):
        H, W, C = image.shape
        orig_size = min(H, W)
        image_size = 128

        image, (xshift, yshift) = resize_and_crop_image(image, image_size)
        preds = self.face_detector.predict_on_image(image)

        if 0 == len(preds):
            return [[]]

        shift = torch.tensor([xshift, yshift] * 2)
        scores = preds[:, -1:]

        # TODO: ugly
        # reverses, x and y to adapt with face-alignment code
        locs = torch.cat(
            (preds[:, 1:2], preds[:, 0:1], preds[:, 3:4], preds[:, 2:3]), dim=1
        )
        bboxlist = torch.cat((locs * orig_size + shift, scores), dim=1)

        return bboxlist

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0


module = torch.jit.script(BlazeFaceDetector())
