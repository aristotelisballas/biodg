import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from commons import BioSignal


class PickleDataset(Dataset):
    def __init__(self, files: list = None, biosignal: BioSignal = None, transforms: list = None):
        self.transforms = transforms
        self.file_list = []
        self.file_list = files
        self.size = len(self.file_list)
        self.biosignal = biosignal

    def __getitem__(self, index):
        # index = index % self.size
        # data_dict: dict = self.file_list[index]
        fp = open(self.file_list[index], 'rb')
        data_dict = pickle.load(fp)
        fp.close()

        data = data_dict['data']
        label = data_dict['label']

        if self.biosignal == BioSignal.ECG:
            data = np.swapaxes(data, 0, 1)
            # label = np.argmax(label)

        if self.biosignal == BioSignal.PCG:
            label = int_to_np(label)
        return data, label

    def __len__(self):
        return self.size


def int_to_np(x: int):
    a = np.zeros(2)
    a[x] = 1

    return a
