import os
from GPT2 import GPTModel
import torch.nn as nn
import torch
from config import BASE_CONFIG, MODEL_CONFIGS
from model_download import download

class ClassificationModel(nn.Module):
    def __init__(self,Model_variant:str='S', num_class:int =2 , num_block2train:int = 2):
        super().__init__()
        self._Model_variant = Model_variant
        model_dir = download(Model_variant)
        weights = torch.load(model_dir, weights_only=True)

        self.config = BASE_CONFIG.update(MODEL_CONFIGS[Model_variant])
        self.model = GPTModel(self.config)
        self.model.load_state_dict(weights)
        self.num_class=num_class
        self.num_block2train = num_block2train
        self._replace_heads()
        self._freez_except()
    def forward(self, x):
        return self.model(x)

    def model_var(self):
        return  self._Model_variant
    def _replace_heads(self):
        self.model.out_head = nn.Linear(self.config['emb_dim'] , self.num_class)

    def _freez_except(self):
        ## Freeze all the model weights
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        ## unfreeze the classification head and the (weights and biases)  layer
        for parameter in self.model.out_head.parameters():
            parameter.requires_grad =True

        ## unfreeze the last layer normalization
        for parameter in self.model.final_norm.parameters():
            parameter.requires_grad = True

        ## unfreeze the self.num_block2train
        for i in range(1,self.num_block2train+1):

            for parameter in self.model.trf_blocks[-i].parameters():
                parameter.requires_grad = True

