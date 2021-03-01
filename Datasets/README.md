# Dataset generation

The scripts in this directory is used to generate the mathematical dataset used for this project.

### Execution
To execute the script you need to run `main.py`.
```
python main.py
```
This will create a TSV file named `infix_dataset.tsv` with the following structure:
```
infix_notation            postfix_notation  evaluation
( ( ( 2 * 9 ) + 9 ) * 9 )	2 9 * 9 + 9 *	243.0
```
The infix_notation column is the mathematical equation in the traditional notation.
The postfix_notation column is the translation into reverse polish (postfix) notation.
And the evaluation column is a sanity check that the postfix notation is correct.