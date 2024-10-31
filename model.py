from models.pix2pixHD.models.pix2pixHD_model import Pix2PixHDModel

from models.VToonify.model.vtoonify import VToonify
from models.pixel2style2pixel.models.psp import pSp

from models.CycleGAN.models import GeneratorResNet
from utils import *

import torch
import torchvision.transforms.functional as F

import torchvision

CARICATURE_MODELS = ["vtoonify", "vtoonify_no_align", "psp"]


class NIR2CARI(torch.nn.Module):
    def __init__(self, caricature_model="vtoonify"):
        super(NIR2CARI, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.caricature_model = (
            caricature_model.lower().strip() if caricature_model else "vtoonify"
        )
        self.load_models()

        if self.caricature_model not in CARICATURE_MODELS:
            print(
                "▶ No caricature models was specified, so only RGB colorization will be performed"
            )

    def load_models(self):
        # pix2pixHD (NIR → RGB 복원 모델) 초기화
        self.pix2pixHD = Pix2PixHDModel()
        self.pix2pixHD.netG.load_state_dict(
            torch.load("models/pix2pixHD/checkpoints/latest_net_G.pth")
        )

        # VToonify (RGB → 캐리커처 변환 모델 1) 초기화
        self.vtoonify = VToonify()
        self.vtoonify.load_state_dict(
            torch.load(
                "models/VToonify/checkpoints/vtoonify_t.pt",
                map_location=lambda storage, loc: storage,
            )["g_ema"],
            strict=False,
        )
        self.vtoonify.to(self.device)

        # pixel2style2pixel (RGB → 캐리커처 변환 모델 2) 초기화
        self.pSp = pSp()
        pSpCheckpoints = torch.load(
            "models/pixel2style2pixel/checkpoints/best_model.pt", map_location="cpu"
        )
        self.pSp.encoder.load_state_dict(
            get_keys(pSpCheckpoints, "encoder"), strict=False
        )
        self.pSp.decoder.load_state_dict(
            get_keys(pSpCheckpoints, "decoder"), strict=False
        )
        self.pSp.eval()
        self.pSp.cuda()

        # CycleGAN (추가 스타일 부여 모델) 초기화
        self.cyclegan = GeneratorResNet((3, 1024, 1024), 9)
        self.cyclegan.cuda()
        self.cyclegan.load_state_dict(
            torch.load("models/CycleGAN/checkpoints/generator.pth")
        )

        print("▶ All models are successfully loaded")

    def forward(self, image: torch.Tensor):
        # NIR → RGB 복원 모델 Inference
        colorized = (
            (self.pix2pixHD(image).squeeze() / 2.0 * 255.0 + 127.0).int().clip(0, 255)
        )
        caricatured = colorized

        # RGB → 캐리커처 변환 모델 Inference
        if self.caricature_model.startswith("vtoonify"):
            # 얼굴 랜드마크 검출 및 입력 영상 정렬 단계 생략 여부
            skip_align = "no_align" in self.caricature_model
            """ 임시 코드
            print("$ import test image")
            colorized = torchvision.io.read_image("test.jpg").to("cuda")
            """
            caricatured = self.vtoonify(colorized, skip_align).squeeze()
        elif self.caricature_model == "psp":
            caricatured = self.pSp(colorized)[0]
        else:
            # 지정된 캐리커처 변환 모델이 없을 경우, Inference 중단 후 RGB Colorized 결과물 반환
            return colorized.permute(1, 2, 0)

        # RGB → YIQ 색 공간 변환 후 Y 채널에 대해서만 Style transfer 적용
        Y, I, Q = yiq_from_image(caricatured)
        luminance = extend_to_three_channel(Y)
        real = luminance.unsqueeze(dim=0)

        stylized = self.cyclegan(real)

        stylized = F.rgb_to_grayscale(stylized)
        R, G, B = yiq_to_rgb(stylized, I, Q)

        # 다시 RGB 색 공간으로 변환 후 채널 병합
        stylized = (
            torch.stack([B.squeeze(), G.squeeze(), R.squeeze()])
            .permute(1, 2, 0)
            .mul(127)
            .add(127)
            .clip(0, 255)
        )

        return stylized
