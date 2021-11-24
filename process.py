"""
    process:
        Refactoring and reshaping the dataset
"""
from configurations import DatasetConfig
from datahelpers import DataRefactorer


if __name__=="__main__":
    config = DatasetConfig().get_args()
    DataRefactorer(config)
