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
- Use descriptive file names
- Add a short docstring or header comment explaining the exercise
- Add a small example command to run the exercise

## License & Contact
- License: (add your license, e.g., MIT)
- Maintainer: jaikrishnaads