import os.path
from .base_dataset import BaseDataset, get_params, get_transform, normalize
from .image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        ### input A (label maps)
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert("RGB"))
        B_tensor = inst_tensor = feat_tensor = 0

        input_dict = {
            "label": A_tensor,
            "inst": inst_tensor,
            "image": B_tensor,
            "feat": feat_tensor,
            "path": A_path,
        }

        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return "AlignedDataset"
