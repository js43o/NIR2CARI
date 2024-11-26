from models.pix2pixHD.data.data_loader import CreateDataLoader
from model import NIR2CARI

import numpy as np
import os
import torch
import argparse
from PIL import Image
from time import time

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataroot", type=str, help="a path where input images exist", default="dataset"
)
parser.add_argument(
    "--output", type=str, help="a path where output images will be", default="output"
)
parser.add_argument(
    "--caricature_model",
    type=str,
    help='which caricature model to be used for initial caricature generation ("vtoonify" | "vtoonify_no_align" | "psp")',
    default="vtoonify",
)
options = vars(parser.parse_args())


if __name__ == "__main__":
    data_loader = CreateDataLoader(options["dataroot"])
    dataset = data_loader.load_data()
    os.makedirs(options["output"], exist_ok=True)

    print("▶ Loading models...")
    nir2cari = NIR2CARI(options["caricature_model"])

    dataset_len = len(dataset)

    for i, data in enumerate(dataset):
        image = data["label"]
        filename = os.path.basename(data["path"][0]).split(".")[0]

        time_prev = time()
        result = nir2cari(image)

        print(
            "▷ %s (%d ms) (%d/%d)"
            % (filename, int((time() - time_prev) * 1000), i + 1, dataset_len)
        )

        result = Image.fromarray(result.detach().cpu().numpy().astype(np.uint8))
        result.save("%s/%s.png" % (options["output"], filename))

    print("▶ Done!")
