"""
classification_models.py

Defines standard and LoRA-adapted GPT-2 based classification models using pre-trained GPT backbones.

Classes:
    - ClassificationModel: Standard fine-tuning on a few transformer blocks and output head.
    - LoRA_Classification_Model: Lightweight LoRA-based fine-tuning for efficient adaptation.

Dependencies:
    - torch
    - GPTModel (custom GPT2 architecture)
    - config (BASE_CONFIG, MODEL_CONFIGS)
    - model_download (download checkpoint function)
    - lorautils (ReplaceLinear for LoRA injection)
"""

from GPT2 import GPTModel
import torch.nn as nn
import torch
from config import BASE_CONFIG, MODEL_CONFIGS
from model_download import download
from lorautils import ReplaceLinear
class ClassificationModel(nn.Module):
    def __init__(self,Model_variant:str='S', num_class:int =2 , num_block2train:int = 2 , Dropout:float =0.0 ):
        """
           GPT-2 based text classification model with partial fine-tuning.

           Args:
               Model_variant (str): Variant of the GPT model to use (e.g., 'S', 'M', 'L').
               num_class (int): Number of output classes.
               num_block2train (int): Number of transformer blocks (from the end) to fine-tune.

           Behavior:
               - Loads pre-trained GPT weights.
               - Replaces the output head with a classification layer.
               - Freezes most layers except the output head, final norm, and last N transformer blocks.

           Example:
               model = ClassificationModel(Model_variant='S', num_class=3, num_block2train=2)
        """


        super().__init__()
        self._Model_variant = Model_variant
        model_dir = download(Model_variant)
        weights = torch.load(model_dir, weights_only=True)
        self.config = BASE_CONFIG
        self.config.update(MODEL_CONFIGS[Model_variant])
        self.config['drop_rate'] = Dropout
        self.model = GPTModel(self.config)
        self.model.load_state_dict(weights)
        self.num_class=num_class
        self.num_block2train = num_block2train
        self._replace_heads()
        self._freez_except()
    def forward(self, x):
        """
                Forward pass through the GPT-based classification model.
        """
        return self.model(x)

    def model_var(self):
        """
               Returns the model variant name.
        """
        return  self._Model_variant
    def _replace_heads(self):
        """
                Replaces the GPT output layer with a classification head.
        """

        self.model.out_head = nn.Linear(in_features=self.config['emb_dim']
                                        , out_features=self.num_class)

    def _freez_except(self):

        """
        Freezes all parameters except:
            - The output classification head
            - The final layer normalization
            - The last `num_block2train` transformer blocks
        """
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




class LoRA_Classification_Model(nn.Module):
    """
        GPT-2 based classification model with LoRA adapters for parameter-efficient fine-tuning.

        Args:
            Model_variant (str): Variant of the GPT model to use (e.g., 'S', 'M', 'L' , 'XL').
            num_class (int): Number of output classes.
            rank (int): LoRA rank.
            alpha (float): LoRA scaling factor.

        Behavior:
            - Loads frozen pre-trained GPT weights.
            - Adds LoRA adapters to attention blocks.
            - Only LoRA-injected layers and output head are trainable.

        Example:
            model = LoRA_Classification_Model(Model_variant='S', num_class=3, rank=8, alpha=2.0)
        """

    def __init__(self,Model_variant:str='S', num_class:int =2 ,rank = 16 , alpha = 1.5 , Dropout:float = .0):
        super().__init__()
        self._Model_variant = Model_variant
        model_dir = download(Model_variant)
        weights = torch.load(model_dir, weights_only=True)
        self.config = BASE_CONFIG
        self.config.update(MODEL_CONFIGS[Model_variant])
        self.config['drop_rate'] = Dropout
        self.model = GPTModel(self.config)
        self.model.load_state_dict(weights)
        self.num_class=num_class
        self._freeze()
        self._replace_out()
        ReplaceLinear(self.model,  rank=rank , alpha = alpha)
    def forward(self , x):
        """
                Forward pass through the LoRA-enhanced GPT-based classification model.
        """
        return self.model(x)


    def _freeze(self):
        """
               Freezes all parameters of the model.
        """

        ## Freeze all the model weights
        for parameter in self.model.parameters():
            parameter.requires_grad = False


    def _replace_out(self):
        """
               Replaces the output head with a classification layer.
        """

        self.model.out_head = nn.Linear(in_features=self.config['emb_dim']
                                        , out_features=self.num_class)