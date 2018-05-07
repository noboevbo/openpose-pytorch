from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __len__(self):
        return NotImplementedError

    def __getitem__(self, index):
        return NotImplementedError

    def get_dataset_id_from_index(self, index):
        return NotImplementedError