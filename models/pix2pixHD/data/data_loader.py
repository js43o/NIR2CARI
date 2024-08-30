from .custom_dataset_data_loader import CustomDatasetDataLoader


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader
