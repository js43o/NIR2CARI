from .pix2pixHD_model import InferenceModel


def create_model(opt):
    model = InferenceModel()
    model.initialize(opt)

    return model
