import json
import os.path
import sys
import pandas as pd
import torch

sys.path.append('src')

from src import (split_data , CreateDataloader,ClassificationModel,train_classifier_ddp,train_df_dir,
                  test_df_dir ,val_df_dir,CONFIG_DICT_DIR,DEVICE ,
                 LoRA_Classification_Model, ClassifierDataset
                 )

def main(json_name):
    path = os.path.join(CONFIG_DICT_DIR , json_name)
    with open(path, "r") as json_file:
        data_config = json.load(json_file)
    csv_dir = data_config['data_dir']
    label_dictionary= data_config["label_dictionary"]
    column_name = data_config['column_name']
    Model_variant = data_config['Model_variant']
    batch_size = data_config['Batch_size']
    max_seq_length = data_config['max_seq_length']
    number_of_class = data_config['number_of_class']
    train_split = data_config['train_split']
    val_split   =  data_config['val_split']
    num_epochs = data_config['num_epochs']
    lr = data_config['lr']
    weight_decay = data_config['weight_decay']
    LoRA = data_config["LoRA"]
    Dropout = data_config["Dropout"]
    if LoRA:
        Rank  = data_config["Rank"]
        alpha = data_config["alpha"]
    else :
        num_layer2train = data_config["num_layer2train"]

    df = pd.read_csv(csv_dir)
    split_data(data = df , label_dict=label_dictionary,
               train_split=train_split,val_split=val_split,column_name=column_name)


    train_dataset = ClassifierDataset(train_df_dir , max_seq_length , column_name = column_name)

    val_dataset= ClassifierDataset(val_df_dir, max_seq_length, column_name=column_name)

    if LoRA :
        model = LoRA_Classification_Model(Model_variant=Model_variant ,
                                          num_class=number_of_class ,
                                          rank=Rank , alpha=alpha , Dropout=Dropout)
        print("ALL of US LOVE LoRA")
        print(type(model))
    else:
        model = ClassificationModel(Model_variant=Model_variant ,num_class=number_of_class,
                                    num_block2train=num_layer2train,Dropout=Dropout)
        print("ALL of US HATE LoRA")
        print(type(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_classifier_ddp(
        local_rank=0,
        world_size=2,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer_fn=optimizer,
        num_epochs=num_epochs)
    #torch.save(model.module.state_dict(), f'final.pt')
    return  model