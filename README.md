# NIR2CARI
A caricature generator from near-infrared (NIR) face images

## 사용된 Base Model 목록
- pix2pixHD (https://github.com/NVIDIA/pix2pixHD)
- **VToonify** (https://github.com/williamyang1991/VToonify)
- CycleGAN (https://github.com/eriklindernoren/PyTorch-GAN)

## 실행 환경
- RTX 3080 Ti (12GB VRAM)
- **CUDA 11.8**
- Python 3.8
- Pytorch 2.0.0

## 환경 세팅 방법
1. `conda create -n nir2cari python=3.8` 명령어로 python 3.8 conda 가상 환경을 생성합니다.
2. `conda activate nir2cari` 명령어로 방금 생성한 가상환경을 활성화합니다.
3. `conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia` 명령어로 pytorch, torchvision 패키지를 설치합니다.
4. 프로젝트 루트 경로에서 `pip install -r requirements.txt` 명령을 통해 나머지 패키지들을 설치합니다.

## 실행 방법 (모델 추론)
1. `dataset` 폴더에 입력 데이터로 사용할 NIR 이미지를 보관합니다.
2. `python run` 명령을 입력하여 모델 추론을 실행합니다.
3. `output` 폴더에 변환된 캐리커처 이미지가 출력됩니다.

- 다음과 같이 인자를 직접 지정하여 실행할 수 있습니다.<br />
  `python run.py --gpu_ids 3 --dataroot "../nir_images" --output "../cari_images"`
