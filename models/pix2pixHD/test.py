import os
from .data.data_loader import CreateDataLoader
from .models.models import create_model
from .util import util

def main(opt):
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    model = create_model(opt)

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        
        generated = model.inference(data["label"], data["inst"], data["image"])
        basename = os.path.basename(data["path"][0])
        filename, extension = os.path.splitext(basename)
        
        util.save_image(util.tensor2im(generated.data[0]), os.path.join(opt.results_dir, filename + '_rgb' + extension))

