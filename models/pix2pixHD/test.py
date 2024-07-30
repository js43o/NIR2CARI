import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website

model = create_model(opt)

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    """
    if opt.data_type == 16:
        data["label"] = data["label"].half()
        data["inst"] = data["inst"].half()
    elif opt.data_type == 8:
        data["label"] = data["label"].uint8()
        data["inst"] = data["inst"].uint8()
    if opt.export_onnx:
        print("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith(
            "onnx"
        ), "Export model file should end with .onnx"
        torch.onnx.export(
            model, [data["label"], data["inst"]], opt.export_onnx, verbose=True
        )
        exit(0)
    minibatch = 1
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data["label"], data["inst"]])
    elif opt.onnx:
        generated = run_onnx(
            opt.onnx, opt.data_type, minibatch, [data["label"], data["inst"]]
        )
    else:
        generated = model.inference(data["label"], data["inst"], data["image"])
    """
    generated = model.inference(data["label"], data["inst"], data["image"])
    basename = os.path.basename(data["path"][0])
    filename, extension = os.path.splitext(basename)
    
    util.save_image(util.tensor2im(generated.data[0]), os.path.join(opt.results_dir, filename + '_rgb' + extension))

