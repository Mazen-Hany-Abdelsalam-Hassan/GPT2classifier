<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>GPT-2 Text Classifier</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 40px;
      line-height: 1.6;
      max-width: 900px;
    }
    code {
      background-color: #f4f4f4;
      padding: 2px 4px;
      border-radius: 4px;
    }
    pre {
      background-color: #f4f4f4;
      padding: 15px;
      border-radius: 5px;
      overflow-x: auto;
    }
  </style>
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
    <li>✅ <strong>Custom training loop</strong> without high-level frameworks</li>
    <li>✅ <strong>Custom GPT-2 classifier head</strong></li>
    <li>✅ Minimal dependencies (focused on core libraries like PyTorch and Hugging Face)</li>
    <li>✅ Clear modular design for easy extension</li>
  </ul>

  <h2>📁 Project Structure</h2>
  <pre>
GPT2classifier/
│
├── classifier_model.py      # GPT-2 classifier model definition
├── config.py                # Hyperparameters and settings
├── dataset.py               # Dataset loading and preprocessing
├── train.py                 # Fully custom training loop
├── evaluate.py              # Manual evaluation script
├── utils.py                 # Helper functions
└── requirements.txt         # Project dependencies
  </pre>

  <h2>⚙️ Setup & Usage</h2>
  <ol>
    <li><strong>Clone the repository</strong>
      <pre><code>git clone https://github.com/Mazen-Hany-Abdelsalam-Hassan/GPT2classifier.git
cd GPT2classifier</code></pre>
    </li>

    <li><strong>Install dependencies</strong>
      <pre><code>pip install -r requirements.txt</code></pre>
    </li>

    <li><strong>Train the model</strong>
      <pre><code>python train.py</code></pre>
    </li>

    <li><strong>Evaluate the model</strong>
      <pre><code>python evaluate.py</code></pre>
    </li>
  </ol>

  <h2>🧠 How It Works</h2>
  <ul>
    <li>Loads a pre-trained GPT-2 model from Hugging Face</li>
    <li>Freezes/Unfreezes layers as needed</li>
    <li>Adds a custom classification head on top of GPT-2 outputs</li>
    <li>Implements the training loop manually using PyTorch</li>
  </ul>

  <h2>📚 Notes</h2>
  <p>
    This project is meant for <strong>learning and experimentation</strong> with custom model training using transformer architectures.
    It avoids using Trainer APIs to give full visibility and control over every training step.
  </p>

</body>
</html>
