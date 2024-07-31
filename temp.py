import os
from PIL import Image

CARI_PATH = '../../datasets/lfw_caricatures'
REAL_PATH = '../../datasets/lfw_real_face'
SAVE_PATH = '../../datasets/lfw_caricatures_hard'

caris = set(map(lambda name: name.split("_vtoonify")[0], os.listdir(CARI_PATH)))
reals = set(map(lambda name: name.split(".")[0], os.listdir(REAL_PATH)))
diff = list(map(lambda name: name + ".jpg", reals.difference(caris)))

print(len(diff))
  