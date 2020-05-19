import torch
from torch.utils.data import DataLoader, Dataset


class DiffusionDataSet(Dataset):
    def __init__(self, features, adj):
        """pytorch dataset

        Arguments:
            features {np.ndarray} -- graph node feature matrix, (N, d)
            adj {sparse tensor} -- graph edge connections
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.adj = adj

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return index


class DiffusionDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers, shuffle=False, drop_last=False):
        # assert shuffle == False # mostly the id for the graph should not change
        self.dataset = dataset
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, drop_last=drop_last)
