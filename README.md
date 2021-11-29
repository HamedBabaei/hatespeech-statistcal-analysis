# Hate Speech 

## Directory Structures

## Classifier 1: `NgramLSVM`

First, train the model to save the pre-trained model, next use the pretrained model for evaluation.

1. To train and save NgramLSVM model (`model hypterparameters avaliable in args, no need to change it.`)

```pythonregexp
python3 ml_runner.py --model_name ml --test False
```
The model will be saved in `pretrained-model/ml/` directory. And 5-fold cross-validation will be performed on train set and logs (results) will be stored in `logs/` directory

2. To test on validation and test sets
```pythonregexp
python3 ml_runner.py --model_name ml --test True
```
The model will store the results on validation and test sets in `logs/` dir.
 


## Classifier 2: `RoBERTa`


## Statistical Analysis

## Requirements

* GPU (for fast training of RoBERTa)
* Python3.9
* Linux
* Python libraries (exist in requirements.txt)

