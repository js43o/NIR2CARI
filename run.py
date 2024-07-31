import subprocess
from models.pix2pixHD.test import main as p2pHD
from options.pix2pixHD import Options as pix2pix_opts


if __name__ == "__main__":
    pix2pix_opt = pix2pix_opts()
    p2pHD(pix2pix_opt)

    subprocess.run(
        [
            "python",
            "models/VToonify/style_transfer.py",
            "--input_path",
            "temp_output",
            "--ckpt",
            "models/VToonify/checkpoint/vtoonify_t.pt",
            "--style_encoder_path",
            "models/VToonify/checkpoint/encoder.pt",
            "--faceparsing_path",
            "models/VToonify/checkpoint/faceparsing.pth",
            "--output_path",
            "output",
            "--scale_image",
        ]
    )
