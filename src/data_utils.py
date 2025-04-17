import os
import pandas as pd
from config import *
import torch
from torch.utils.data import Dataset

def split_data(data: pd.DataFrame, train_split: int = .8, val_split: int = .10):
    os.makedirs(PARENT_DIR, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)
    data_size = len(data)
    train_size = int(train_split * data_size)
    val_size = int((train_split + val_split) * data_size)
    train_df = data[:train_size]
    val_df = data[train_size: val_size]
    test_df = data[val_size:]
    train_df.iloc[::, 1] = train_df.iloc[::, 1].map(LABEL_DICTIONARY)
    val_df.iloc[::, 1] = val_df.iloc[::, 1].map(LABEL_DICTIONARY)
    test_df.iloc[::, 1] = test_df.iloc[::, 1].map(LABEL_DICTIONARY)
    train_df.to_csv(train_df_dir, index=False)
    val_df.to_csv(val_df_dir, index=False)
    test_df.to_csv(test_df_dir, index=False)


class ClassifierDataset(Dataset):
    def __init__(self, csv_dir, tokenizer):
        super().__init__()
        data = pd.read_csv(csv_dir)
        self.X = data.loc[::, 'review'].apply(tokenizer.encode)
        self.Y = data.loc[::, 'sentiment']

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        # print(x)
        length = len(x)
        diff = ALLOWED_SEQ_LENGTH - length
        if diff >= 0:
            padding = diff * [PADDINGTOKEN]
            x = x + padding
        else:
            x = x[:ALLOWED_SEQ_LENGTH]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.X)
