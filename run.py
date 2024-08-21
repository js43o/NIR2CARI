from models.pix2pixHD.test import main as p2pHD
from models.VToonify.style_transfer import main as vtoonify
from models.CycleGAN.inference import main as cyclegan
from options.pix2pixHD import Options as pix2pix_opts
from options.vtoonify import Options as vtoonify_opts

import os

os.makedirs("output/pix2pixHD", exist_ok=True)
os.makedirs("output/vtoonify", exist_ok=True)
os.makedirs("output/cyclegan", exist_ok=True)

if __name__ == "__main__":
    pix2pix_opt = pix2pix_opts()
    # p2pHD(pix2pix_opt)

    vtoonify_opt = vtoonify_opts()
    vtoonify(vtoonify_opt)

    cyclegan()
