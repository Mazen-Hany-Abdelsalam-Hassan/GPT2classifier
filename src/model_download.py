import os
import urllib
from config import  (CLASSIFICATION_MODEL_DIR,
                     BASE_MODELS_DIR,
                     DATA_PATH ,
                     MODEL_PATHS
                     )
def download(Model_variant:str = 'S'):

    os.makedirs(BASE_MODELS_DIR , exist_ok=True)
    os.makedirs(CLASSIFICATION_MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    keys = MODEL_PATHS.keys()
    if Model_variant not in keys:
        raise KeyError(f"Invalid model variant: {Model_variant}. Available keys: {keys}")


    model_name = MODEL_PATHS[Model_variant]
    save_dir = os.path.join(BASE_MODELS_DIR,model_name)
    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{model_name}"
    if not os.path.exists(save_dir):
        urllib.request.urlretrieve(url, save_dir)
        print(f"Downloaded to {save_dir}")
    return save_dir



