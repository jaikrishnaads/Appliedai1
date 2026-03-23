# AppliedAI1

A collection of small exercises, scripts, and notebooks for learning core AI/ML concepts.

## Table of Contents
- [About](#about)
- [Exercises](#exercises)
- [How to run the exercises](#how-to-run-the-exercises)
- [Project layout](#project-layout)
- [Recommendations](#recommendations)
- [Contributing](#contributing)
- [License & Contact](#license--contact)

## About
This repository contains compact, hands-on examples demonstrating foundational topics like logic gates and perceptrons, forward and backward propagation, activation functions, and a small sample dataset for experiments. The exercises are meant for learning and experimentation.

## Exercises
The repository currently includes the following exercise files. Each entry shows the file name, location, short description, and suggested run/open steps.

- ex2andandor.py — Root
  - Description: Small Python script demonstrating logical operations (AND / OR) and a simple perceptron example. Likely contains example inputs, outputs, and a simple training/update loop.
  - How to run: `python ex2andandor.py`
  - Notes: If the script prints results or plots, run it in a terminal or an environment that supports plotting.

- ex3forwardandbackwardpropagation.py — Root
  - Description: Script that demonstrates forward and backward propagation on a tiny neural network or on a toy example. Useful to study gradients, weight updates, and loss computation step-by-step.
  - How to run: `python ex3forwardandbackwardpropagation.py`
  - Notes: Inspect the top of the file for any required dependencies or hard-coded data paths.

- ex3sigmod,relu,tanx.ipynb — Root (Jupyter notebook)
  - Description: Notebook exploring activation functions: sigmoid, ReLU, tanh (filename currently contains commas). Contains plots, comparisons of derivatives, and example usage in networks.
  - How to open: Launch with Jupyter: `jupyter notebook ex3sigmod,relu,tanx.ipynb` or open in VS Code/JupyterLab.
  - Notes: Consider renaming to `ex3_activation_functions.ipynb` to avoid commas and spaces.

- Untitled (1).ipynb — Root (Jupyter notebook)
  - Description: Additional notebook — contents not documented. Likely an ad-hoc or scratch notebook used during experimentation.
  - How to open: `jupyter notebook "Untitled (1).ipynb"`
  - Notes: Rename to a descriptive name and add a short header cell describing the notebook's purpose.

- heart.csv — Root (data file)
  - Description: Small dataset (heart disease / medical features) used for classification experiments or demos.
  - How to use: Load with pandas: `pd.read_csv("heart.csv")`. Check the notebooks or scripts for reference to expected columns or preprocessing steps.

If you want, I can:
- Add brief README subsections that embed short excerpts (1–2 lines) from each script/notebook describing function names, main classes, or variables.
- Open and extract the first markdown/header cell of each notebook to include its brief summary into the README.
- Rename files to consistent, descriptive names (I recommend avoiding spaces and commas in filenames).

## How to run the exercises
1. Create a virtual environment and install packages (if any):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt  # if you add one
   ```
2. Run Python scripts:
   ```bash
   python ex2andandor.py
   python ex3forwardandbackwardpropagation.py
   ```
3. Open notebooks:
   ```bash
   jupyter notebook
   # then open the .ipynb files from the browser UI
   ```

## Project layout
- ex2andandor.py
- ex3forwardandbackwardpropagation.py
- ex3sigmod,relu,tanx.ipynb
- Untitled (1).ipynb
- heart.csv
- README.md

## Recommendations
- Rename notebooks/files to remove spaces/commas. Suggested names:
  - ex2_and_or.py
  - ex3_forward_backward_propagation.py
  - ex3_activation_functions.ipynb
  - notebook_explorations.ipynb
- Add a requirements.txt with libraries used (numpy, pandas, matplotlib, jupyter, etc.).
- Add a 1–2 line header cell to each notebook describing its goal; I can automatically extract and insert those descriptions into README if you want.
- If any scripts expect specific data paths or packages, add brief usage examples and parameter descriptions under each exercise entry.

## Contributing
Contributions welcome. Please open an issue or send a pull request. When contributing exercises:
# Applied AI Laboratory (U23ADS603) - Lab Record & Repository

**Register Number:** 61782323110046  
**Course:** Applied AI Laboratory  
**Department:** Artificial Intelligence and Data Science  
**Academic Year:** 2025-2026

---

## Table of Contents
1. [About](#about)
2. [Experiment 6: Optimization Techniques in DNN](#experiment-6-optimization-techniques-in-dnn)
3. [Experiment 7.1: Gradient-Based Optimization Analysis](#experiment-71-gradient-based-optimization-analysis)
4. [Experiment 7.2: Visual Analysis on MNIST](#experiment-72-visual-analysis-on-mnist)
5. [Experiment 9.1: Sentiment Analysis Using RNN](#experiment-91-sentiment-analysis-using-rnn)
6. [Experiment 9.2: Neural Machine Translation Using Transformer](#experiment-92-neural-machine-translation-using-transformer)
7. [Installation & Setup](#installation--setup)
8. [Requirements](#requirements)
9. [License](#license)

---

## About
This repository contains the implementation, analysis, and records for the Applied AI Laboratory course. It covers foundational AI concepts including optimization algorithms, deep neural networks, recurrent neural networks (RNN), and transformer models. Each experiment includes objective, algorithm specification, dataset details, implementation, and results.

---

## Experiment 6: Optimization Techniques in DNN

**Date:** 09/02/2026  
**Course Outcomes:** CO1

### Objective
To implement and compare different gradient-based optimization techniques on a deep neural network using the Bank Marketing dataset, analyze their convergence, stability, and performance, and identify the optimizer that achieves the best generalization on unseen data.

### Algorithm Specification
- **Algorithm:** Deep Neural Network (DNN)
- **Architecture:** Input → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)
- **Loss Function:** Binary Cross-Entropy
- **Optimizers:** Batch GD, SGD, Mini-Batch, Momentum, Nesterov
- **Data Split:** 70% Training, 15% Validation, 15% Testing
- **Scaling:** StandardScaler

### Dataset Specification
- **Dataset:** Bank Marketing Dataset
- **Records:** 41,188 client records
- **Features:** 20 input features (categorical and numerical)
- **Target:** Binary prediction (Yes/No subscription)

### Key Observations
1. Mini-batch Gradient Descent showed faster and smoother convergence compared to Batch and SGD.
2. SGD had more fluctuations in loss due to frequent weight updates.
3. Momentum and Nesterov optimizers improved convergence speed and reduced oscillations.
4. Dropout helped reduce overfitting by keeping training and validation loss closer.
5. Early stopping prevented unnecessary training once validation loss stopped improving.

### Result
The deep neural network was trained using multiple optimization techniques on the Bank Marketing dataset. Momentum and Nesterov optimizers showed faster convergence and better stability, while Mini-Batch provided efficient training. The best optimizer achieved high accuracy and demonstrated good generalization on the test data.

---

## Experiment 7.1: Gradient-Based Optimization Analysis

**Date:** 16/02/2026  
**Course Outcomes:** CO2

### Objective
To implement and visually analyze gradient-based optimization algorithms and compare their convergence behavior using an analytical function (Himmelblau's Function) and a pose classification dataset.

### Algorithm Specification
- **Function:** $J(w) = (w_1^2 + w_2 - 11)^2 + (w_1 + w_2^2 - 7)^2$
- **Optimizers:** GD, Nesterov, Adagrad, RMSProp, Adam
- **Visualization:** 2D Contour plots, 3D Surface plots, Loss curves

### Key Observations
1. Basic Gradient Descent showed slow convergence and oscillations in curved regions.
2. Momentum-based methods (Nesterov) reached minima faster with smoother trajectories.
3. Adaptive optimizers (Adam, RMSProp, Adagrad) converged quickly and handled steep and flat regions effectively.
4. Adam optimizer provided the most stable and fastest convergence among all methods.

### Result
Optimization algorithms differ significantly in convergence speed and stability. Adam achieves the best overall performance due to its combined momentum and adaptive learning capabilities.

---

## Experiment 7.2: Visual Analysis on MNIST

**Date:** 16/02/2026  
**Course Outcomes:** CO2

### Objective
To implement and compare different gradient-based optimization algorithms on a real-world image dataset (MNIST) using a logistic regression model without bias, and to visually analyze optimizer convergence behavior using animated plots.

### Algorithm Specification
- **Model:** Logistic Regression (No Bias Term)
- **Features:** Mean and Standard Deviation of pixel intensity (2 features)
- **Output:** Softmax (10 classes)
- **Loss:** Categorical Cross-Entropy
- **Optimizers:** GD, SGD, Mini-Batch, Nesterov, Adagrad, RMSProp, Adam

### Key Observations
1. Gradient Descent showed smooth but relatively slower convergence.
2. SGD exhibited noticeable fluctuations due to stochastic updates.
3. Adam achieved faster and smoother convergence compared to other optimizers.
4. Adaptive methods adjusted step sizes dynamically.

### Result
The experiment confirms that gradient-based optimization algorithms exhibit distinct convergence behaviors when applied to a real-world multi-class image classification problem. Adam provides the most efficient and consistent performance on the MNIST dataset.

---

## Experiment 9.1: Sentiment Analysis Using RNN

**Date:** 16/03/2026  
**Course Outcomes:** CO2

### Objective
To implement a stacked Recurrent Neural Network (RNN) model for binary sentiment classification on the IMDB movie review dataset, and to evaluate the model's performance using accuracy/loss curves, confusion matrix, F1 score, classification report, and ROC-AUC curve.

### Model Specification
- **Model Type:** Stacked SimpleRNN with Embedding Layer
- **Vocabulary Size:** 10,000
- **Sequence Length:** 100 tokens
- **Layers:** Embedding → SimpleRNN(64) → SimpleRNN(64) → Dense(64) → Output(1)
- **Optimizer:** Adam (lr=0.001, clipnorm=1.0)
- **Callbacks:** EarlyStopping, ReduceLROnPlateau

### Dataset Specification
- **Dataset:** IMDB Movie Reviews
- **Total Samples:** 50,000 (25,000 train + 25,000 test)
- **Classes:** 2 (Negative, Positive)

### Key Observations
1. The model achieved a test accuracy of approximately 68.58% and a test loss of 0.6047.
2. F1 Score: 0.7176, indicating a fair balance between precision and recall.
3. Adaptive learning rate strategy through ReduceLROnPlateau was crucial for enabling the model to converge.
4. SimpleRNN models are susceptible to vanishing gradients, limiting their ability to capture long-range dependencies.

### Result
The experiment confirms that a stacked SimpleRNN model with an embedding layer can learn sequential patterns for binary sentiment classification. The model achieved a test accuracy of 68.58% and an F1 score of 0.7176 on the IMDB dataset.

---

## Experiment 9.2: Neural Machine Translation Using Transformer

**Date:** 16/03/2026  
**Course Outcomes:** CO2

### Objective
To implement an interactive English-to-French Neural Machine Translation system using the pre-trained Helsinki-NLP/opus-mt-en-fr MarianMT Transformer model, evaluate its translation quality using the BLEU metric on the OPUS100 parallel corpus, and demonstrate batch and interactive translation capabilities.

### Model Specification
- **Model:** Helsinki-NLP/opus-mt-en-fr (MarianMT)
- **Tokenizer:** MarianTokenizer
- **Decoding:** Beam Search (num_beams=5)
- **Evaluation:** BLEU Score (sacrebleu)
- **Dataset:** OPUS100 (English-French)

### Key Observations
1. BLEU Score on 500 test sentences: 14.75.
2. The model successfully translated technical terms and multi-word NLP phrases.
3. The model maintained grammatical correctness and proper French gender agreement.
4. Interactive translation interface allowed real-time testing.

### Result
The experiment successfully demonstrated the deployment of a pre-trained MarianMT Transformer model for English-to-French machine translation. A BLEU score of 14.75 was achieved on 500 test sentences from the OPUS100 dataset.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AppliedAI1
- Use descriptive file names
- Add a short docstring or header comment explaining the exercise
- Add a small example command to run the exercise

## License & Contact
- License: (add your license, e.g., MIT)
- Maintainer: jaikrishnaads
