<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
</head>
<body>

  <h1>GPT-2 Text Classifier (Implemented from Scratch)</h1>

  <p>
    This project demonstrates how to build a <strong>text classification system using GPT-2</strong>,
    implemented <strong>entirely from scratch</strong> using PyTorch and Hugging Face's GPT-2 model as a base.
    While GPT-2 is commonly used for language generation, this project repurposes it for classification
    by adding a custom classification head and training logic.
  </p>

  <h2>📌 Key Features</h2>
  <ul>
    <li>✅ Custom training and evaluation logic (no Trainer APIs)</li>
    <li>✅ GPT-2 model adapted for classification tasks</li>
    <li>✅ Modular and clean code design</li>
    <li>✅ Easy to understand and extend for learning purposes</li>
  </ul>
 <h2>📁 Project Structure</h2>
  <pre>
GPT2classifier/
│         
├── src/                              # Source code
│   ├── __init__.py                   # Init file for the src package
│   ├── config.py                     # Hyperparameters and configuration
│   ├── create_config_file.py         # Dynamic config file generator
│   ├── data_utils.py                 # Text data preprocessing utilities
│   ├── GPT2.py                       # GPT-2 model wrapper
│   ├── GPT2Modification.py           # GPT-2 classifier head modifications
│   ├── inference.py                  # Prediction logic
│   ├── lorautils.py                  # LoRA utilities (if used for fine-tuning)
│   ├── model_download.py             # Model downloading utilities
│   └── train_evaluate.py             # Training and evaluation functions
│
├── main.py                           # Main script to run prediction
└── requirements.txt                  # Python dependencies
</pre>
  <h2>📓 Usage</h2>
  <p>
    All training and evaluation steps are provided in dedicated Jupyter Notebooks.
    Simply open the notebooks and follow the instructions inside to run the model.
  </p>

  <h2>🧠 How It Works</h2>
  <ul>
    <li>Loads a pre-trained GPT-2 model from Hugging Face</li>
    <li>Adds a classification head on top of the transformer output</li>
    <li>Prepares text datasets for supervised training</li>
    <li>Implements custom loss calculation, optimizer updates, and evaluation metrics</li>
  </ul>

  <h2>📚 Notes</h2>
  <p>
    This project was developed for educational purposes and shows how transformer-based models like GPT-2
    can be adapted beyond their original use cases. It provides a transparent and hands-on understanding of training deep learning models for classification.
    
    The version of the package will not affect at all , The code is pure pytorch
    
  </p>

</body>
</html>
