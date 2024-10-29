import torch.utils.data
from .base_data_loader import BaseDataLoader


def CreateDataset(dataroot: str):
    dataset = None
    from .aligned_dataset import AlignedDataset

    dataset = AlignedDataset()

    # print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(dataroot)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return "CustomDatasetDataLoader"

    def initialize(self, dataroot: str):
        BaseDataLoader.initialize(self, dataroot)
        self.dataset = CreateDataset(dataroot)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
