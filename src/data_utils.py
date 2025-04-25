"""
Data Utilities for GPT-2 Classification
=======================================

This module provides utility functions and classes for preparing data to train GPT-2 as a sequence classifier.
It includes functionality for tokenizing, batching with dynamic sequence lengths, and dataset management.

Functions:
----------

- dynamic_batch_loader:
    A collate function for dynamically padding sequences within a batch.

- sort_df:
    Tokenizes and sorts a DataFrame by sequence length to optimize batching efficiency.

- CreateDataloader:
    Constructs a PyTorch DataLoader using the ClassifierDataset and a custom collate function.

- split_data:
    Splits a given DataFrame into train/val/test CSVs and optionally maps labels using a dictionary.

Classes:
--------

- ClassifierDataset:
    A PyTorch Dataset that reads tokenized text data from a CSV, truncates sequences, and appends an EOS token.
"""




import pandas as pd
from config import *
import os
import torch
from torch.utils.data import Dataset , DataLoader

def dynamic_batch_loader(batch: list, padding_token=PADDINGTOKEN):
    """
       Custom collate function for dynamically padding sequences in a batch.

       Args:
           batch (list): List of (tokenized_sequence, label) pairs.
           padding_token: Token used to pad sequences to equal length.

       Returns:
           Tuple[torch.Tensor, torch.Tensor]: Padded sequences and corresponding labels.
       """


    max_len = max(len(x) for x, _ in batch)

    X, Y = [], []
    for x, y in batch:
        padded_x = x + [padding_token] * (max_len - len(x))
        X.append(padded_x)
        Y.append(y)

    return torch.tensor(X, dtype=torch.long), torch.tensor(Y)


def sort_df(data:pd.DataFrame, column_name:list=COLUMN_NAME_IMDB, tokenizer = TOKENIZER):
    """
       Tokenizes and sorts a DataFrame by the length of each sequence.

       Args:
           data (pd.DataFrame): DataFrame with text and label columns.
           column_name (list): List with [text_column_name, label_column_name].
           tokenizer: Tokenizer with an .encode() method.

       Returns:
           pd.DataFrame: Sorted DataFrame with 'encoded_text' and 'length' columns.
       """


    data.loc[::,'encoded_text'] = data.loc[::,column_name[0]].apply(tokenizer.encode)
    data.loc[::,'length'] = data.loc[::,'encoded_text'].apply(len)
    sorted_df = data.sort_values("length")
    return sorted_df.reset_index( drop=True)



class ClassifierDataset(Dataset):
    """
        PyTorch Dataset for classification using GPT-2.

        Args:
            csv_dir (str): Path to the CSV file.
            allowed_seq_length (int): Max allowed sequence length.
            column_name (list): [text_column_name, label_column_name].
            tokenizer: Tokenizer with .encode() method.

        Returns:
            (List[int], Any): Tokenized and truncated sequence with appended PAD token, and its label.
        """


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
    """
       Creates a DataLoader for classification training.

       Args:
           csv_dir (str): Path to the CSV file.
           batch_size (int): Number of samples per batch.
           allowed_seq_length (int): Max input sequence length.
           column_name (list): List with [text_column_name, label_column_name].
           tokenizer: Tokenizer used to encode the text.
           collate_fn: Function for collating batch samples.

       Returns:
           DataLoader: Configured PyTorch DataLoader.
       """

    dataset = ClassifierDataset(csv_dir=csv_dir , allowed_seq_length=allowed_seq_length,
                                tokenizer=tokenizer,column_name=column_name)
    dataloader = DataLoader(dataset , batch_size=batch_size , collate_fn=collate_fn,shuffle=False)
    return  dataloader



def split_data(data: pd.DataFrame, label_dict = LABEL_DICTIONARY_IMDB,
               column_name = COLUMN_NAME_IMDB,
               train_split: float = .8, val_split: float = .10, seed = SEED):

    """
    Splits dataset into train, validation, and test sets, applies label mapping, and saves CSV files.

    Args:
        data (pd.DataFrame): Full dataset as a DataFrame.
        label_dict (dict): Optional dictionary to map label names to integers.
        column_name (list): [text_column_name, label_column_name].
        train_split (float): Proportion of training data.
        val_split (float): Proportion of validation data.
        seed (int): Random seed for reproducibility.

    Saves:
        train.csv, val.csv, and test.csv to predefined directory paths.
    """



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
