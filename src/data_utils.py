import pandas as pd
from config import *
import os
import torch
from torch.utils.data import Dataset , DataLoader

def dynamic_batch_loader(batch: list, padding_token=PADDINGTOKEN):
    max_len = max(len(x) for x, _ in batch)

    X, Y = [], []
    for x, y in batch:
        padded_x = x + [padding_token] * (max_len - len(x))
        X.append(padded_x)
        Y.append(y)

    return torch.tensor(X, dtype=torch.long), torch.tensor(Y)


def sort_df(data:pd.DataFrame,tokenizer = TOKENIZER):
    data.loc[::,'encoded_text'] = data.loc[::,'review'].apply(tokenizer.encode)
    data.loc[::,'length'] = data.loc[::,'encoded_text'].apply(len)
    sorted_df = data.sort_values("length")
    return sorted_df.reset_index( drop=True)



class ClassifierDataset(Dataset):
    def __init__(self, csv_dir, allowed_seq_length = ALLOWED_SEQ_LENGTH,tokenizer = TOKENIZER):
        super().__init__()
        self.data = pd.read_csv(csv_dir)
        self.data = sort_df(self.data , tokenizer = tokenizer)
        self.X = self.data.loc[::, 'encoded_text']
        self.Y = self.data.loc[::, 'sentiment']
        self.allowed_seq_length = allowed_seq_length
    def __getitem__(self, idx):
        x = self.X[idx]
        x = x[:self.allowed_seq_length-1]
        x.append(PADDINGTOKEN)
        y = self.Y[idx]
        return x , y
    def __len__(self):
        return len(self.X)

def CreateDataloader(csv_dir, batch_size = 16,
                      allowed_seq_length = ALLOWED_SEQ_LENGTH,
                      tokenizer = TOKENIZER,
                      collate_fn=dynamic_batch_loader):
    dataset = ClassifierDataset(csv_dir=csv_dir , allowed_seq_length=allowed_seq_length,
                                tokenizer=tokenizer)
    dataloader = DataLoader(dataset , batch_size=batch_size , collate_fn=collate_fn)
    return  dataloader



def split_data(data: pd.DataFrame, train_split: int = .8, val_split: int = .10 , seed = SEED,label_dict = LABEL_DICTIONARY):
    os.makedirs(PARENT_DIR, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    data_size = len(data)
    train_size = int(train_split * data_size)
    val_size = int((train_split + val_split) * data_size)
    train_df = data[:train_size]
    val_df = data[train_size: val_size]
    test_df = data[val_size:]
    train_df.loc[::, 'sentiment'] = train_df.loc[::, 'sentiment'].map(LABEL_DICTIONARY)
    val_df.loc[::, 'sentiment'] = val_df.loc[::, 'sentiment'].map(LABEL_DICTIONARY)
    test_df.loc[::, 'sentiment'] = test_df.loc[::, 'sentiment'].map(LABEL_DICTIONARY)
    train_df.to_csv(train_df_dir, index=False)
    val_df.to_csv(val_df_dir, index=False)
    test_df.to_csv(test_df_dir, index=False)
