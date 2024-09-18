from models.pix2pixHD.data.data_loader import CreateDataLoader
from model import NIR2CARI

import os
import torch
import argparse
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu_ids", type=str, help="which gpu when going inference", default="0"
)
parser.add_argument(
    "--dataroot", type=str, help="path where input images exist", default="dataset"
)
parser.add_argument(
    "--output", type=str, help="path where output images will be", default="output"
)
options = vars(parser.parse_args())
options["gpu_ids"] = list(map(lambda x: int(x), options["gpu_ids"].split(",")))


if __name__ == "__main__":
    data_loader = CreateDataLoader(options)
    dataset = data_loader.load_data()
    os.makedirs(options["output"], exist_ok=True)

    nir2cari = NIR2CARI(options)
    times = []

    for i, data in enumerate(dataset):
        image = data["label"]
        filename = os.path.basename(data["path"][0]).split(".")[0]
        # time_s = time.time()
        nir2cari(image, filename)
        # times.append(time.time() - time_s)

    # print("avg =", sum(times) / len(times))
