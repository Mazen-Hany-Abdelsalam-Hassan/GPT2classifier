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


def sort_df(data:pd.DataFrame, column_name:list=COLUMN_NAME_IMDB, tokenizer = TOKENIZER):
    data.loc[::,'encoded_text'] = data.loc[::,column_name[0]].apply(tokenizer.encode)
    data.loc[::,'length'] = data.loc[::,'encoded_text'].apply(len)
    sorted_df = data.sort_values("length")
    return sorted_df.reset_index( drop=True)



class ClassifierDataset(Dataset):
    def __init__(self, csv_dir,
                 allowed_seq_length = ALLOWED_SEQ_LENGTH,
                 column_name:list = COLUMN_NAME_IMDB,
                 tokenizer = TOKENIZER):
        super().__init__()
        self.data = pd.read_csv(csv_dir)
        self.data = sort_df(self.data ,column_name=column_name, tokenizer = tokenizer)
        self.X = self.data.loc[::, 'encoded_text']
        self.Y = self.data.loc[::, column_name[1]]
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
                     column_name:list = COLUMN_NAME_IMDB,
                     tokenizer = TOKENIZER,
                     collate_fn=dynamic_batch_loader):
    dataset = ClassifierDataset(csv_dir=csv_dir , allowed_seq_length=allowed_seq_length,
                                tokenizer=tokenizer,column_name=column_name)
    dataloader = DataLoader(dataset , batch_size=batch_size , collate_fn=collate_fn,shuffle=False)
    return  dataloader



def split_data(data: pd.DataFrame, label_dict = LABEL_DICTIONARY_IMDB,
               column_name = COLUMN_NAME_IMDB,
               train_split: float = .8, val_split: float = .10, seed = SEED):
    os.makedirs(PARENT_DIR, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    data_size = len(data)
    train_size = int(train_split * data_size)
    val_size = int((train_split + val_split) * data_size)
    train_df = data[:train_size]
    val_df = data[train_size: val_size]
    test_df = data[val_size:]
    train_df.loc[::, column_name[1]] = train_df.loc[::, column_name[1]].apply(lambda x : x if  label_dict is None else label_dict[x])
    val_df.loc[::, column_name[1]] = val_df.loc[::, column_name[1]].apply(lambda x : x if  label_dict is None else label_dict[x])
    test_df.loc[::, column_name[1]] = test_df.loc[::, column_name[1]].apply(lambda x : x if  label_dict is None else label_dict[x])
    train_df.to_csv(train_df_dir, index=False)
    val_df.to_csv(val_df_dir, index=False)
    test_df.to_csv(test_df_dir, index=False)
