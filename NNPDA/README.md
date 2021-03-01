# Neural Network Pushdown Automata

This directory contains our work in implementing a NNPDA.
The `orig.py` file contains the [original code](https://github.com/rajeev595/NNPDA) by user rajeev595.
However, this file is neither up to date with current library versions, nor is it runnable.
We use this work as a starting point for our own implementation in TensorFlow 2. 

Our implementation can be found in the `nnpda.py` file.
The `NNPDA Cheat Sheet.pdf` file provides the list of notation used in both the paper and the code and explains what the notation means.
Our implementation of a NNPDA entails the creation of a custom RNN cell that performs the logic described by Sun et al. [CoRR '93].
i.e., it contains logic for equation 5 and 23/24 of the paper, which detail how to get the next state and action.

There remains to be implemented the training of a neural network based on our custom RNN cell.

### Files
- The `nnpda.py` file contains our current work in implementing the NNPDA.

- `orig.py` is the original code by user rajeev595 that we use as a starting point and guide for our own implementation.

- `stack.py` contains an implementation of a Stack that will be used in the NNPDA

- `rnn.py` is the code following an online guide. This was used to confirm our understanding of a RNN, which provided insight into how the NNPDA should function.

### Execution
This file can be run with 
```
python nnpda.py
```
As of now, no concrete data is used as the implementation of the NNPDA is not yet complete.