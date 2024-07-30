import os
import random
from PIL import Image

nir_faces = os.listdir("../../datasets/lfw_nir_face")
rgb_faces = os.listdir("../../datasets/lfw_real_face")
names = list(map(lambda x: x.split(".")[0], nir_faces))

random.shuffle(names)

for idx, name in enumerate(names):
    typeof = "train" if idx < len(names) * 0.9 else "test"
    Image.open(os.path.join("../../datasets/lfw_nir_face", name + ".png")).save(
        os.path.join("../../datasets/nir_to_rgb", typeof + "_A", name + ".jpg")
    )
    Image.open(os.path.join("../../datasets/lfw_real_face", name + ".jpg")).save(
        os.path.join("../../datasets/nir_to_rgb", typeof + "_B", name + ".jpg")
    )
