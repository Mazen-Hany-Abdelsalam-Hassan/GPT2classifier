import os
import tiktoken
import torch
MODEL_PATHS = {
              "S" : "gpt2-small-124M.pth",
              "M" : "gpt2-medium-355M.pth",
              "L" : "gpt2-large-774M.pth",
              "XL": "gpt2-xl-1558M.pth"              }

BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

MODEL_CONFIGS = {
    "S": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "L": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "XL": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_SEQ_LENGTH = 1000
SEED = 123

LABEL_DICTIONARY_IMDB = {"positive": 1 , "negative":0}
COLUMN_NAME_IMDB = ["review" , "sentiment"]
PARENT_DIR= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PARENT_DIR ,'data')
train_df_dir = os.path.join(DATA_PATH, "train.csv")
val_df_dir = os.path.join(DATA_PATH, "val.csv")
test_df_dir = os.path.join(DATA_PATH, "test.csv")
TOKENIZER = tiktoken.encoding_for_model('gpt-2')
PADDINGTEXT = "<|endoftext|>"
PADDINGTOKEN = TOKENIZER.encode(PADDINGTEXT,allowed_special = 'all')[0]
BASE_MODELS_DIR = os.path.join(PARENT_DIR, 'BASEModel')
CLASSIFICATION_MODEL_DIR = os.path.join(PARENT_DIR, 'CLASSIFICATION')
CONFIG_DICT_DIR = os.path.join(PARENT_DIR,'config')





