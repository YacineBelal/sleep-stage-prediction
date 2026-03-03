from torch.utils.data import Dataset


class DreamtDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
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