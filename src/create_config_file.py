import json

import os
from config import *


def create_json(
        data_dir:str="IMDB.csv",
        config_name:str='IMDB',
        label_dictionary:dict=LABEL_DICTIONARY_IMDB,
        column_name:list=COLUMN_NAME_IMDB,
        Model_variant:str='S',
        Batch_size:int=16,
        max_seq_length:int=ALLOWED_SEQ_LENGTH,
        train_split:float = .8,
        val_split:float= .1,
        lr:float= 5e-5,
        weight_decay:float= .1
):
    number_of_class = len(label_dictionary)
    os.makedirs(CONFIG_DICT_DIR,exist_ok=True)
    config_dict = {
        "data_dir":data_dir,
        "label_dictionary":label_dictionary,
        "column_name":column_name,
        "Model_variant":Model_variant,
        "Batch_size":Batch_size,
        "max_seq_length":max_seq_length,
        "number_of_class":number_of_class,
        "train_split": train_split,
        "val_split":val_split,
        "lr":lr,
        "weight_decay":weight_decay }
    save_dir = os.path.join(CONFIG_DICT_DIR,config_name+'.json')
    with open(save_dir, "w") as json_file:
        json.dump(config_dict, json_file, indent=4)
