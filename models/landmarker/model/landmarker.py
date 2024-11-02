from torch import nn

from models.landmarker.model.detection.blazeface import FaceDetector
from models.landmarker.model.utils import *


class Landmarker(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self.face_detector = FaceDetector(device=self.device)
        self.face_alignment_net = torch.jit.load(
            load_file_from_url("models/landmarker/checkpoints/2DFAN4-cd938726ad.zip")
        )
        self.face_alignment_net.to(self.device, dtype=torch.float32)
        self.face_alignment_net.eval()

    def forward(self, x):
        detected_faces = self.face_detector.detect_from_image(x)

        if detected_faces.shape[1] == 0:
            print("# No faces were detected.")
            return None

        landmarks = []

        for i, d in enumerate(detected_faces):
            center = torch.stack(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0]
            )
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale
            inp = crop(x, center, scale.item())
            inp = inp.permute(2, 0, 1)

            inp = inp.to(self.device, dtype=torch.float32)
            inp.div_(255.0).unsqueeze_(0)
            out = self.face_alignment_net(inp).detach()

            pts, pts_img, scores = get_preds_fromhm(out, center, scale.item())

            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
            scores = scores.squeeze(0)

            landmarks.append(pts_img)

        return landmarks
