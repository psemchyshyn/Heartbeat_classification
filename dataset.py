import pandas as pd
import torch

from torch.utils.data import Dataset

class MITBIH_DATASET(Dataset):
    def __init__(self, path, val=False, test=False, split=0.8):
        self.path = path
        self.test = test
        self.val = val
        self.split = split

        self.df = pd.read_csv(path, header=None)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        if not self.test:
            if not self.val:
                self.df = self.df.iloc[:int(len(self.df)*self.split)]
            else:
                self.df = self.df.iloc[int(len(self.df)*self.split):]


    def __getitem__(self, idx):
        x = self.df.iloc[idx].values
        return torch.tensor(x[: -1]).unsqueeze(-1).float(), torch.tensor(x[-1]).long()

    def __len__(self):
        return len(self.df)
