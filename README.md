# AppliedAI1

A short description of the AppliedAI1 project — a collection of exercises and notebooks for learning core AI/ML concepts.

## Table of contents
- [About](#about)
- [Exercises](#exercises)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## About
This repository contains small exercises, example scripts, and notebooks used to explore foundational applied AI topics (activation functions, forward/backward propagation, simple logical/Perceptron examples, and a small dataset for experiments).

## Exercises
The repository currently contains the following exercise files and resources:

- [ex2andandor.py](ex2andandor.py) — Python script demonstrating logical operations (AND / OR) and simple perceptron logic.
- [ex3forwardandbackwardpropagation.py](ex3forwardandbackwardpropagation.py) — Python script that illustrates forward and backward propagation for a small neural network or example model.
- [ex3sigmod,relu,tanx.ipynb](ex3sigmod,relu,tanx.ipynb) — Jupyter notebook exploring activation functions (sigmoid, ReLU, tanh) and their properties. (Filename contains commas; open directly in the repo.)
- [Untitled (1).ipynb](Untitled%20(1).ipynb) — Additional notebook (rename and document as needed).
- [heart.csv](heart.csv) — Small dataset (heart disease / medical features) used for exercises and examples.

If you want, I can add short descriptions inside each notebook/script or rename files to clearer names (recommended: avoid commas and spaces).

## Installation
1. Clone the repo:
   git clone https://github.com/jaikrishnaads/Appliedai1.git
2. Create and activate a virtual environment (recommended):
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate     # Windows
3. Install dependencies (if you have a requirements file):
   pip install -r requirements.txt

## Usage
- Open notebooks in the `notebooks/` directory (or click the `.ipynb` files in the repo) using Jupyter or VS Code.
- Run scripts directly, for example:
   python ex2andandor.py
   python ex3forwardandbackwardpropagation.py

## Development
- Run linting and tests (if present):
   pip install -r requirements-dev.txt
   pytest

## Contributing
Contributions are welcome — please open an issue to discuss changes or create a pull request. Consider adding a CONTRIBUTING.md for contribution guidelines.

## License
Add your license name here (e.g., MIT). If you want, I can add a LICENSE file too.
