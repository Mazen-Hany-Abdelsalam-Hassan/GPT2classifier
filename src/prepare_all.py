import  os
import urllib
from config import MODEL_DIR , BASE_MODELS_DIR ,  CLASSIFICATION_MODEL_DIR,DATA_PATH
def download():
    save_dir = os.path.join(BASE_MODELS_DIR,MODEL_DIR)
    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{MODEL_DIR}"
    if not os.path.exists(save_dir):
        urllib.request.urlretrieve(url, save_dir)
        print(f"Downloaded to {save_dir}")

def main():
    os.makedirs(BASE_MODELS_DIR , exist_ok=True)
    os.makedirs(CLASSIFICATION_MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    download()

if __name__ == '__main__':
    main()

