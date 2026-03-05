import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda


class DreamtDataset(Dataset):
    DEFAULT_TRANSFORM = Compose(
        [
            torch.FloatTensor,
            Lambda(lambda x: x.permute([1, 0])),
        ]
    )

    def __init__(self, X, y, transform=DEFAULT_TRANSFORM, target_transform=None):
        super().__init__()
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        if self.transform:
            x = self.transform(self.X[index])

        if self.target_transform:
            y = self.target_transform(self.y[index])

        return x, y

    def __len__(self):
        return self.X.shape[0]