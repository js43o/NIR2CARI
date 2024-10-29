from models.pix2pixHD.data.data_loader import CreateDataLoader
from model import NIR2CARI

import numpy as np
import os
import torch
import argparse
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataroot", type=str, help="path where input images exist", default="dataset"
)
parser.add_argument(
    "--output", type=str, help="path where output images will be", default="output"
)
options = vars(parser.parse_args())


if __name__ == "__main__":
    data_loader = CreateDataLoader(options)
    dataset = data_loader.load_data()
    os.makedirs(options["output"], exist_ok=True)

    nir2cari = NIR2CARI(options)

    for i, data in enumerate(dataset):
        image = data["label"]
        filename = os.path.basename(data["path"][0]).split(".")[0]

        result = nir2cari(image)

        result = Image.fromarray(result.detach().cpu().numpy().astype(np.uint8))
        result.save("%s/%s.png" % (options["output"], filename))
