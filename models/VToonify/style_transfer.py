import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import numpy as np
import cv2
import dlib
import torch
from torchvision import transforms
import torch.nn.functional as F
from model.vtoonify import VToonify
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import (
    save_image,
    load_psp_standalone,
    get_video_crop_parameter,
)


class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Style Transfer")
        self.parser.add_argument(
            "--content",
            type=str,
            default=None,
            help="path of the content image/video",
        )
        self.parser.add_argument(
            "--input_path",
            type=str,
            default=None,
            help="path of the input images",
        )
        self.parser.add_argument(
            "--ckpt",
            type=str,
            default="./checkpoint/vtoonify_t.pt",
            help="path of the saved model",
        )
        self.parser.add_argument(
            "--output_path",
            type=str,
            default="./output/",
            help="path of the output images",
        )
        self.parser.add_argument(
            "--scale_image",
            action="store_true",
            help="resize and crop the image to best fit the model",
        )
        self.parser.add_argument(
            "--style_encoder_path",
            type=str,
            default="./checkpoint/encoder.pt",
            help="path of the style encoder",
        )
        self.parser.add_argument(
            "--faceparsing_path",
            type=str,
            default="./checkpoint/faceparsing.pth",
            help="path of the face parsing model",
        )
        self.parser.add_argument(
            "--cpu", action="store_true", help="if true, only use cpu"
        )
        self.parser.add_argument(
            "--padding",
            type=int,
            nargs=4,
            default=[200, 200, 200, 200],
            help="left, right, top, bottom paddings to the face center",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="batch size of frames when processing video",
        )
        self.parser.add_argument(
            "--parsing_map_path",
            type=str,
            default=None,
            help="path of the refined parsing map of the target video",
        )

    def parse(self):
        self.opt = self.parser.parse_args()
        # args = vars(self.opt)
        # print("Load options")
        # for name, value in sorted(args.items()):
        #    print("%s: %s" % (str(name), str(value)))
        return self.opt


if __name__ == "__main__":
    parser = TestOptions()
    args = parser.parse()

    device = "cpu" if args.cpu else "cuda"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    vtoonify = VToonify()
    vtoonify.load_state_dict(
        torch.load(args.ckpt, map_location=lambda storage, loc: storage)["g_ema"]
    )
    vtoonify.to(device)

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(
        torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage)
    )
    parsingpredictor.to(device).eval()

    modelname = "models/VToonify/checkpoint/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(modelname):
        import wget, bz2

        wget.download(
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            modelname + ".bz2",
        )
        zipfile = bz2.BZ2File(modelname + ".bz2")
        data = zipfile.read()
        open(modelname, "wb").write(data)
    landmarkpredictor = dlib.shape_predictor(modelname)

    pspencoder = load_psp_standalone(args.style_encoder_path, device)

    # print("Load models successfully!")

    filelist = os.listdir(args.input_path) if args.input_path else [args.content]
    for _filename in filelist:
        filename = (
            os.path.join(args.input_path, _filename) if args.input_path else _filename
        )
        basename = os.path.basename(filename).split(".")[0]
        scale = 1
        kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])
        
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        
        cropname = os.path.join(args.output_path, basename + "_input.jpg")
        savename = os.path.join(
            args.output_path, basename + "_vtoonify" + ".jpg"
        )

        frame = cv2.imread(filename)
        print(filename)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
        # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
        if args.scale_image:
            paras = get_video_crop_parameter(frame, landmarkpredictor, args.padding)
            if paras is not None:
                h, w, top, bottom, left, right, scale = paras
                H, W = int(bottom - top), int(right - left)
                # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
                if scale <= 0.75:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                if scale <= 0.375:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

        with torch.no_grad():
            I = align_face(frame, landmarkpredictor)
            if I is None:  
                print("Face landmarks was NOT detected, so skip translation of this image")
                continue
                
            I = transform(I).unsqueeze(dim=0).to(device)
            s_w = pspencoder(I)
            s_w = vtoonify.zplus2wplus(s_w)

            x = transform(frame).unsqueeze(dim=0).to(device)
            # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
            # followed by downsampling the parsing maps
            x_p = F.interpolate(
                parsingpredictor(
                    2
                    * (
                        F.interpolate(
                            x, scale_factor=2, mode="bilinear", align_corners=False
                        )
                    )
                )[0],
                scale_factor=0.5,
                recompute_scale_factor=False,
            ).detach()
            # we give parsing maps lower weight (1/16)
            inputs = torch.cat((x, x_p / 16.0), dim=1)
            # d_s has no effect when backbone is toonify
            y_tilde = vtoonify(
                inputs, s_w.repeat(inputs.size(0), 1, 1)
            )
            y_tilde = torch.clamp(y_tilde, -1, 1)

        cv2.imwrite(cropname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        save_image(y_tilde[0].cpu(), savename)

    # print("Transfer style successfully!")
