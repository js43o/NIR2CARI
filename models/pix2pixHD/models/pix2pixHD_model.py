import numpy as np
import torch
from torch import nn
from .base_model import BaseModel
from . import networks
import warnings
from typing import Optional
from torch import Tensor
import os
import sys

warnings.filterwarnings("ignore", "volatile")


class Pix2PixHDModel(nn.Module):
    @torch.jit.export
    def name(self):
        return "Pix2PixHDModel"

    def __init__(self):
        self.opt = {}
        self.gpu_ids = []
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join("models/pix2pixHD/checkpoints")
        self.use_features = False
        self.gen_features = False
        self.netG = None

    @torch.jit.export
    def initialize(self, opt):
        # BaseModel.initialize(self, opt)
        self.opt = opt
        self.gpu_ids = opt["gpu_ids"]

        torch.backends.cudnn.benchmark = True

        ##### define networks
        # Generator network
        netG_input_nc = 3
        self.netG = networks.define_G(
            netG_input_nc, 3, 64, "global", 4, 9, 1, 3, "instance", gpu_ids=self.gpu_ids
        )

        # load networks
        self.load_network(self.netG, "G", "latest")

    @torch.jit.export
    def encode_input(
        self,
        label_map,
        inst_map=None,
        real_image=None,
        feat_map: Optional[Tensor] = None,
        infer: bool = False,
    ):
        input_label = label_map.data.cuda()

        # real images for training
        if real_image is not None:
            real_image = real_image.data.cuda()

        return input_label, inst_map, real_image, feat_map

    def forward(self, label, inst, image, feat, infer: bool = False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(
            label, inst, image, feat, infer
        )

        # Fake Generation
        input_concat = input_label
        fake_image = self.netG.forward(input_concat)

        return fake_image

    @torch.jit.export
    def inference(self, label, inst, image=None):
        # Encode Inputs
        input_label, inst_map, real_image, _ = self.encode_input(
            label, inst, image, None, True
        )

        # Fake Generation
        input_concat = input_label

        if torch.__version__.startswith("0.4"):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=""):
        save_filename = "%s_net_%s.pth" % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            if network_label == "G":
                raise ("Generator must exist!")
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {
                        k: v for k, v in pretrained_dict.items() if k in model_dict
                    }
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print(
                            "Pretrained network %s has excessive layers; Only loading layers that are used"
                            % network_label
                        )
                except:
                    print(
                        "Pretrained network %s has fewer layers; The following are not initialized:"
                        % network_label
                    )
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        from sets import Set

                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if (
                            k not in pretrained_dict
                            or v.size() != pretrained_dict[k].size()
                        ):
                            not_initialized.add(k.split(".")[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)


@torch.jit.export
class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp[0], inp[1]
        return self.inference(label, inst)
