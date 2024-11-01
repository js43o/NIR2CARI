import torch
import math
from torch import nn
import torch.nn.functional as F

from .bisenet.model import BiSeNet
from .stylegan.model import Generator

from ..model.encoder.encoders.psp_encoders import GradualStyleEncoder
from models.landmarker.model.landmarker import Landmarker
from torchvision.transforms import functional as TF


def load_psp_standalone(checkpoint_path, device="cuda"):
    psp = GradualStyleEncoder()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    psp_dict = {
        k.replace("encoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("encoder.")
    }

    psp.load_state_dict(psp_dict)
    psp.eval().to(device)

    return psp


class VToonifyResBlock(nn.Module):
    def __init__(self, fin):
        super().__init__()

        self.conv = nn.Conv2d(fin, fin, 3, 1, 1)
        self.conv2 = nn.Conv2d(fin, fin, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        out = self.lrelu(self.conv2(out))
        out = (out + x) / math.sqrt(2)
        return out


class VToonify(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 얼굴 랜드마크 검출 모델 초기화
        self.landmarkpredictor = Landmarker()

        # 얼굴 파싱 모델 초기화
        self.parsingpredictor = BiSeNet(n_classes=19)
        self.parsingpredictor.load_state_dict(
            torch.load(
                "models/VToonify/checkpoints/faceparsing.pth",
                map_location=lambda storage, loc: storage,
            )
        )
        self.parsingpredictor.to(self.device).eval()

        # 스타일 벡터 Encoder 초기화
        self.pspencoder = load_psp_standalone(
            "models/VToonify/checkpoints/encoder.pt", self.device
        )

        # Encoder 초기화
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3 + 19, 32, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(32, 128, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(256, 256, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 512, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 512, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ),
                nn.Sequential(
                    VToonifyResBlock(512),
                    VToonifyResBlock(512),
                    VToonifyResBlock(512),
                    VToonifyResBlock(512),
                    VToonifyResBlock(512),
                    VToonifyResBlock(512),
                ),
                nn.Conv2d(512, 3, 1, 1, 0, bias=True),
            ]
        )

        # Decoder 초기화 (StyleGAN2 Generator)
        self.generator = Generator()

        # Fusion 모듈 초기화
        self.fusion_out = nn.ModuleList(
            [
                nn.Conv2d(1024, 512, 3, 1, 1, bias=True),
                nn.Conv2d(1024, 512, 3, 1, 1, bias=True),
                nn.Conv2d(512, 256, 3, 1, 1, bias=True),
                nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            ]
        )
        self.fusion_skip = nn.ModuleList(
            [
                nn.Conv2d(515, 3, 3, 1, 1, bias=True),
                nn.Conv2d(515, 3, 3, 1, 1, bias=True),
                nn.Conv2d(259, 3, 3, 1, 1, bias=True),
                nn.Conv2d(131, 3, 3, 1, 1, bias=True),
            ]
        )

    def forward(self, x: torch.Tensor, skip_align: bool = False):
        """주어진 입력 얼굴 영상을 상응하는 캐리커처 영상으로 변환합니다.

        Args:
            x (torch.Tensor): 입력 얼굴 영상
            skip_align (bool, optional): 얼굴 랜드마크 검출 및 입력 영상 정렬 생략 여부

        Returns:
            torch.Tensor: 출력 캐리커처 영상
        """
        x = x / 255.0

        if not skip_align:
            # 입력 영상에서 얼굴 랜드마크 검출을 통해 크롭에 필요한 인자들을 계산
            paras = self.get_video_crop_parameter(x.permute(1, 2, 0))

            if paras is not None:
                h, w, top, bottom, left, right, scale = paras
                # 고해상도 영상에 대해 Over-sharpening 현상 방지를 위한 가우시안 블러 적용
                if scale <= 0.75:
                    x = TF.gaussian_blur(x, [3, 3], [0.5, 0.5])

                x = TF.resize(x, [int(h), int(w)], antialias=True)[
                    :, top:bottom, left:right
                ]  # Crop

        with torch.no_grad():
            I = (
                x.clone().detach().to(self.device)
            )  # '스타일 코드'로 인코딩 될 영상 Tensor의 복제본 생성
            if not skip_align:
                # 얼굴 랜드마크 검출 결과를 바탕으로 리사이징, 크롭, 패딩 등 다양한 전처리 적용
                I = self.align_face(x.permute(1, 2, 0))

            I = ((I - 0.5) / 0.5).unsqueeze(0).to(self.device)

            # pSp 인코더를 통해 스타일 코드로 한 차례 변환
            s_w = self.pspencoder(I)

            # 기존 잠재 공간 Z+에서 새로운 잠재 공간 W+로 변환 (StyleGAN2)
            s_w = self.zplus2wplus(s_w)

            x = ((x - 0.5) / 0.5).unsqueeze(0).to(self.device)

            # 얼굴 구조 변형 안정화를 위한 입력 얼굴 영상의 파싱 맵 생성
            x_p = F.interpolate(
                self.parsingpredictor(
                    F.interpolate(
                        x, scale_factor=2.0, mode="bilinear", align_corners=False
                    )
                    * 2
                )[0],
                scale_factor=0.5,
                recompute_scale_factor=False,
            ).detach()

            feat = torch.cat((x, x_p / 16.0), dim=1)  # Featrue Tensor
            styles = s_w.repeat(feat.size(0), 1, 1)  # Style Tensor

        # Encoding 시작
        feat_0 = self.encoder[0](feat)
        feat_1 = self.encoder[1](feat_0)
        feat_2 = self.encoder[2](feat_1)
        feat_3 = self.encoder[3](feat_2)
        feat_4 = self.encoder[4](feat_3)

        out = feat_4
        skip = self.encoder[5](feat_4)

        # Decoding 시작
        f_E = feat_3
        out = self.fusion_out[0](torch.cat([out, f_E], dim=1))
        skip = self.fusion_skip[0](torch.cat([skip, f_E], dim=1))
        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )

        out = self.generator.convs[6](out, styles[:, 7])

        out = self.generator.convs[7](out, styles[:, 8], noise=noise)
        skip = self.generator.to_rgbs[3](out, styles[:, 9], skip)

        f_E = feat_2
        out = self.fusion_out[1](torch.cat([out, f_E], dim=1))
        skip = self.fusion_skip[1](torch.cat([skip, f_E], dim=1))
        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.generator.convs[8](out, styles[:, 9], noise=noise)
        out = self.generator.convs[9](out, styles[:, 10], noise=noise)
        skip = self.generator.to_rgbs[4](out, styles[:, 11], skip)

        f_E = feat_1
        out = self.fusion_out[2](torch.cat([out, f_E], dim=1))
        skip = self.fusion_skip[2](torch.cat([skip, f_E], dim=1))
        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.generator.convs[10](out, styles[:, 11], noise=noise)
        out = self.generator.convs[11](out, styles[:, 12], noise=noise)
        skip = self.generator.to_rgbs[5](out, styles[:, 13], skip)

        f_E = feat_0
        out = self.fusion_out[3](torch.cat([out, f_E], dim=1))
        skip = self.fusion_skip[3](torch.cat([skip, f_E], dim=1))
        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.generator.convs[12](out, styles[:, 13], noise=noise)
        out = self.generator.convs[13](out, styles[:, 14], noise=noise)
        skip = self.generator.to_rgbs[6](out, styles[:, 15], skip)

        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.generator.convs[14](out, styles[:, 15], noise=noise)
        out = self.generator.convs[15](out, styles[:, 16], noise=noise)
        skip = self.generator.to_rgbs[7](out, styles[:, 17], skip)

        image = skip

        return image

    def zplus2wplus(self, zplus):
        return self.generator.style(
            zplus.reshape(zplus.shape[0] * zplus.shape[1], zplus.shape[2])
        ).reshape(zplus.shape)

    ########## image alignment methods ##########

    def get_landmark(self, img):
        landmarks = self.landmarkpredictor(img * 255.0)
        if landmarks is None:
            return None

        return landmarks[0]

    def align_face(self, img):
        """입력 얼굴 영상을 상응하는 스타일 코드(s_w)로 인코딩하기 위한 전처리 함수로써,
        적절한 리사이징과 크롭 등을 수행한 뒤 반사(Reflection) 패딩을 추가하여 반환합니다.

        Args:
            img (torch.Tensor[h, w, c]):

        Returns:
            img (torch.Tensor[h, w, c]):
        """
        lm = self.get_landmark(img)
        if lm is None:
            return img

        lm = lm.detach().cpu()

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
        x /= torch.hypot(x[0], x[1])
        x *= torch.max(
            torch.hypot(eye_to_eye[0], eye_to_eye[1]) * 2.0,
            torch.hypot(eye_to_mouth[0], eye_to_mouth[1]) * 1.8,
        )
        y = torch.flipud(x) * torch.tensor([-1, 1])
        c = eye_avg + eye_to_mouth * 0.1
        quad = torch.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = torch.hypot(x[0], x[1]) * 2

        output_size = 256

        # Shrink.
        shrink = int(torch.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (
                int(torch.round(float(img.shape[0]) / shrink)),
                int(torch.round(float(img.shape[1]) / shrink)),
            )
            img = TF.resize(img, rsize)
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
            int(torch.max(crop[0] - border, torch.tensor(0)).item()),  # left
            int(torch.max(crop[1] - border, torch.tensor(0)).item()),  # top
            int(
                torch.min(crop[2] + border, torch.tensor(img.shape[0])).item()
            ),  # right
            int(
                torch.min(crop[3] + border, torch.tensor(img.shape[1])).item()
            ),  # bottom
        )
        if crop[2] - crop[0] < img.shape[0] or crop[3] - crop[1] < img.shape[1]:
            img = TF.crop(
                img.permute(2, 0, 1),
                crop[1],
                crop[0],
                crop[3],
                crop[2],
            ).permute(1, 2, 0)
            quad -= torch.tensor([crop[1], crop[0]])

        # Pad.
        pad = (
            int(torch.floor(torch.min(quad[:, 0]))),
            int(torch.floor(torch.min(quad[:, 1]))),
            int(torch.ceil(torch.max(quad[:, 0]))),
            int(torch.ceil(torch.max(quad[:, 1]))),
        )
        pad = torch.stack(
            [
                torch.max(-pad[0] + border, torch.tensor(0)),
                torch.max(-pad[1] + border, torch.tensor(0)),
                torch.max(pad[2] - img.shape[0] + border, torch.tensor(0)),
                torch.max(pad[3] - img.shape[1] + border, torch.tensor(0)),
            ],
        )
        if torch.max(pad) > border - 4:
            pad = torch.maximum(pad, torch.round(qsize * 0.3))
            img = TF.pad(
                img.permute(2, 0, 1),
                (
                    int(pad[0].item()),
                    int(pad[1].item()),
                    int(pad[2].item()),
                    int(pad[3].item()),
                ),
                padding_mode="reflect",
            )

            _, h, w = img.shape
            y = torch.arange(h).view(-1, 1, 1)  # Shape: (h, 1, 1)
            x = torch.arange(w).view(1, -1, 1)  # Shape: (1, w, 1)
            mask = (
                torch.maximum(
                    1.0
                    - torch.minimum(
                        x.float() / pad[0],
                        (w - 1 - x).float() / pad[2],
                    ),
                    1.0
                    - torch.minimum(
                        y.float() / pad[1],
                        (h - 1 - y).float() / pad[3],
                    ),
                )
                .permute(2, 0, 1)
                .to("cuda")
            )

            blur = float(qsize * 0.02)
            radius = int(4.0 * blur)
            img = torch.clip(img * 255.0 + 1, 0.0, 255.0)
            img += (
                TF.gaussian_blur(
                    img,
                    kernel_size=[2 * radius + 1, 2 * radius + 1],
                    sigma=[blur, blur],
                )
                - img
            ) * torch.clip(mask * 3.0 + 1.0, 0.0, 1.0)

            q = torch.tensor(0.5).to("cuda")
            img += (
                torch.quantile(torch.quantile(img, q, dim=0), q, dim=0) - img
            ) * torch.clip(mask, 0.0, 1.0)
            img = torch.clip(torch.round(img) / 255.0, 0.0, 1.0)
            quad += pad[:2]

        # Transform.
        img = TF.resize(img, (output_size, output_size), antialias=True)
        # img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)

        return img

    def get_video_crop_parameter(self, img):
        """입력된 얼굴 영상으로부터 각 얼굴 랜드마크의 좌표를 추론한 뒤,
        적절히 크롭하기 위해 필요한 7개의 크롭 인자 값들을 반환합니다.

        Args:
            img (Tensor[c, h, w]):

        Returns:
            coords (Tuple(int, int, int, int, int, int, float)): h, w, top, bottom, left, right, scale
        """
        lm = self.get_landmark(img)
        if lm is None:
            return None

        lm_eye_left = lm[36:42]  # left-clockwise
        lm_eye_right = lm[42:48]  # left-clockwise
        padding = [200, 200, 200, 200]

        scale = 64.0 / (torch.mean(lm_eye_right[:, 0]) - torch.mean(lm_eye_left[:, 0]))
        center = (
            (torch.mean(lm_eye_right, dim=0) + torch.mean(lm_eye_left, dim=0)) / 2
        ) * scale
        h = torch.round(img.shape[0] * scale).int()
        w = torch.round(img.shape[1] * scale).int()

        left = (
            torch.max(torch.round(center[0] - padding[0]), torch.tensor(0)).int().item()
            // 8
            * 8
        )
        right = torch.min(torch.round(center[0] + padding[1]), w).int().item() // 8 * 8
        top = (
            torch.max(torch.round(center[1] - padding[2]), torch.tensor(0)).int().item()
            // 8
            * 8
        )
        bottom = torch.min(torch.round(center[1] + padding[3]), h).int().item() // 8 * 8

        return h.item(), w.item(), top, bottom, left, right, scale
