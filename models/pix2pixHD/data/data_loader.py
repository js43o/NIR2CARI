from .custom_dataset_data_loader import CustomDatasetDataLoader


def CreateDataLoader(dataroot: str):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(dataroot)
    return data_loader
