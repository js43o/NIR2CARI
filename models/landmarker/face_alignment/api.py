import torch
from enum import IntEnum
from packaging import version

from .utils import *


import models.landmarker.face_alignment.detection.blazeface as face_detector_module


class LandmarksType(IntEnum):
    """Enum class defining the type of landmarks to detect.

    ``TWO_D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``TWO_HALF_D`` - this points represent the projection of the 3D points into 3D
    ``THREE_D`` - detect the points ``(x,y,z)``` in a 3D space

    """

    TWO_D = 1
    TWO_HALF_D = 2
    THREE_D = 3


class NetworkSize(IntEnum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4


default_model_urls = {
    "2DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip",
    "3DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip",
    "depth": "https://www.adrianbulat.com/downloads/python-fan/depth-6c4283c0e0.zip",
}

models_urls = {
    "1.6": {
        "2DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.6-c827573f02.zip",
        "3DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.6-ec5cf40a1d.zip",
        "depth": "https://www.adrianbulat.com/downloads/python-fan/depth_1.6-2aa3f18772.zip",
    },
    "1.5": {
        "2DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.5-a60332318a.zip",
        "3DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.5-176570af4d.zip",
        "depth": "https://www.adrianbulat.com/downloads/python-fan/depth_1.5-bc10f98e39.zip",
    },
}


class FaceAlignment:
    def __init__(
        self,
        landmarks_type,
        network_size=NetworkSize.LARGE,
        device="cuda",
        dtype=torch.float32,
        flip_input=False,
        face_detector="sfd",
        face_detector_kwargs=None,
        verbose=False,
    ):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose
        self.dtype = dtype

        if version.parse(torch.__version__) < version.parse("1.5.0"):
            raise ImportError(
                f"Unsupported pytorch version detected. Minimum supported version of pytorch: 1.5.0\
                            Either upgrade (recommended) your pytorch setup, or downgrade to face-alignment 1.2.0"
            )

        network_size = int(network_size)
        pytorch_version = torch.__version__
        if "dev" in pytorch_version:
            pytorch_version = pytorch_version.rsplit(".", 2)[0]
        else:
            pytorch_version = pytorch_version.rsplit(".", 1)[0]

        if "cuda" in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        # face_detector_module = __import__('face_alignment.detection.' + face_detector,
        #                                   globals(), locals(), [face_detector], 0)
        face_detector_kwargs = face_detector_kwargs or {}
        self.face_detector = face_detector_module.FaceDetector(
            device=device, verbose=verbose, **face_detector_kwargs
        )

        # Initialise the face alignemnt networks
        if landmarks_type == LandmarksType.TWO_D:
            network_name = "2DFAN-" + str(network_size)
        else:
            network_name = "3DFAN-" + str(network_size)
        print(pytorch_version, default_model_urls, network_name)
        self.face_alignment_net = torch.jit.load(
            load_file_from_url(
                models_urls.get(pytorch_version, default_model_urls)[network_name]
            )
        )

        self.face_alignment_net.to(device, dtype=dtype)
        self.face_alignment_net.eval()
