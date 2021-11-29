# Hate Speech 

## Directory Structures

```pythonregexp

+configurations/            >> project config files 
   * __init__.py                  > to make imports easier
   * base_config.py               > models configs
   * dataset_config.py            > datahelpers configs
   * test_config.py               > statistical test configs
+datahandlers/              >> data reader and writers
    * __init__.py
    * datareader.py               > data reader methods
    * datawriter.py               > data writer methods
+datahelpers/               >> raw dataset modifier for models
    * __init__.py     
    * data_refactorer.py          > main for data modifiers
    * twitter_refactorer.py       > for our dataset (train, val, test splits)
+datasets/                 >> datasets for the project
    * twitter                     > our dataset
        - raw                        > raw dataset wihtout modification
        - intermediate               > train, val, test sets dir 
+imagse/                   >> images and plots
+logs/                     >> model outputs and results
+models/                   >> models scripts
    * __init__.py
    * evaluation.py                 > evaluation metrics
    * ngram_ml.py                   > modularized NgramLSVM model
    * ngram_ml_utils.py             > NgramLSVM interface
    * roberta_ft.py                 > modularized RoBERTa model for fine-tuning and evaluation setups
    * roberta_ft_utils.py           > RoBERTa interface
+notebooks/
+statistical_tests/

```

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

