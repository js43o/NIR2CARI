# NIR2CARI
A caricature generator from near-infrared (NIR) face images

## 사용된 Base Model 목록
- pix2pixHD (https://github.com/NVIDIA/pix2pixHD)
- **VToonify** (https://github.com/williamyang1991/VToonify)
- CycleGAN (https://github.com/eriklindernoren/PyTorch-GAN)

## 실행 환경
- RTX 3080 Ti (12GB VRAM) 또는 RTX 4090 (24GB VRAM)
- **CUDA 11.8**
- Python 3.8
- Pytorch 2.0.0

## 환경 세팅 방법
1. `conda create -n nir2cari python=3.8` 명령어로 python 3.8 conda 가상 환경을 생성합니다.
2. `conda activate nir2cari` 명령어로 방금 생성한 가상환경을 활성화합니다.
3. 프로젝트 루트 경로에서 `pip install -r requirements.txt` 명령을 통해 나머지 패키지들을 설치합니다.

## 실행 방법 (기본 인자 기준)
### Inference
1. `dataset` 폴더에 입력 데이터로 사용할 NIR 이미지를 삽입합니다.
2. `python run` 명령을 입력하여 모델 추론을 실행합니다.
3. `output` 폴더에 변환된 캐리커처 이미지가 출력됩니다.

### 실행 인자
- `--dataroot`: 입력 NIR 얼굴 영상 파일이 포함된 디렉토리 경로
- `--output`: 출력 캐리커처 얼굴 영상 파일을 저장할 디렉토리 경로
- `--caricature_model`: 캐리커처 초안 생성 모델로 어떤 베이스 모델을 사용할지 지정
  - `"vtoonify"`: **VToonify**, 상업적 라이센스 존재, 얼굴 랜드마크 검출 및 입력 영상 정렬 과정 포함 (높은 품질, 낮은 속도) (**default**)
  - `"vtoonify_no_align"`: **VToonify**, 상업적 라이센스 존재, 얼굴 랜드마크 검출 및 입력 영상 정렬 과정 생략 (높은 속도, 높은 품질(정렬된 영상일 경우) 또는 낮은 품질(정렬되지 않은 영상일 경우))
  - `"psp"`: **pixel2style2pixel**, 라이센스 제약 없음 (높은 속도, 보통 품질(정렬된 영상일 경우) 또는 낮은 품질(정렬되지 않은 영상일 경우))
- **실행 예시**
  `python run.py --dataroot ./nir_images --output ./cari_images caricature_model psp`
