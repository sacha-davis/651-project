# CMPUT 651 Project
By: Sarah Davis, Delaney Lothian, and Henry Tang

The objective of this project is to explore a solution for the transduction problem.
This can be described as the task to not only accept or reject a given string, but additionally translate the string into another language.
In other words, transduction is the combination of recognition with the task of generation.

## Final Submission Related
### Output Files
For machine outputs, graphs, and models that were used in the final paper, please see the output_files folder.

### Code
The code in this repository is incomplete. The large majority of the coding working can be found [here](https://github.com/Hk-tang/marnns), as we built of the fork of another repository.

### Human Readable logs
For our google sheet of human readable logs, click [here](https://docs.google.com/spreadsheets/d/1f-DIOjS_gD20Rv61FrMdxo7keDZ9iRB74MPuOa80TTY/edit?usp=sharing)

## Introduction

This repository contains the code, instructions, and data for our project.

The `Datasets` directory contains the scripts to generate our mathematical equations dataset.
This script generates simple mathematical equations (operators: +,-,/,*,(,)) and its reverse polish notation equivalent.

The `NNPDA` directory contains the (work in progress) code and instructions for the implementation of a neural network pushdown automata (NNPDA).
This algorithm is an important part of our approach for solving the transduction problem.

The `supervised_approaches` directory contains the code and instructions for a sequence-to-sequence implementation.
This algorithm is used to understand how seq2seq can be used for this project, as well as give an idea of what we need to incorporate into the NNPDA code.

The `output_files` directory contains the results from each of the project parts.
Currently, only the seq2seq model outputs to this directory, containing the results of translating from French to English and infix to postfix mathematical notation.

## Installation and execution

1. Activate a virtual environment and install dependencies 
	```bash
    python3 -m venv venv
 
    venv\Scripts\activate.bat  # windows
    source venv/bin/activate  # unix
 
    pip install -r requirements.txt
	```
 
2. Generate the mathematical equations dataset
   ```bash
   cd Datasets
   python main.py
   ```
 
3. Run the part of the project you are interested in.
    More details can be found in the README.md files of each project part.

 ## Data
 
 `Datasets/infix_dataset.tsv` is the training data we generated and `Datasets/infix_dataset_test.tsv` is our test set.
