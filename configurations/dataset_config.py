"""
    DatasetConfig
         Get dataset parameter for modifications
"""
import argparse
from pathlib import Path
import os


class DatasetConfig:
    """
        Dataset Configigurations
    """
    def __init__(self):
        """
            Dataset configuration
        """
        dataset = "twitter" #[twitter, pan2021]
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--raw_train_dir", type=str,
                                 default=os.path.join(Path(__file__).parents[1].__str__(),
                                                      f"datasets/{dataset}/raw"),
                                 help='Path to root directory')
        self.parser.add_argument("--intermediate_train_dir", type=str,
                                 default=os.path.join(Path(__file__).parents[1].__str__(),
                                                          f"datasets/{dataset}/intermediate"),
                                 help='Path to intermediate directory')
        self.parser.add_argument("--processed_train_dir", type=str,
                                 default=os.path.join(Path(__file__).parents[1].__str__(),
                                                      f"datasets/{dataset}/processed"),
                                 help='Path to processed directory')
        self.parser.add_argument("--logs_dir", type=str,
                                 default=os.path.join(Path(__file__).parents[1].__str__(), "logs"),
                                 help='Path to logs directory')
        self.parser.add_argument("--dataset", type=str,
                                 default=dataset,
                                 help='Dataset name')

    def get_args(self):
        """
            Return parser
        :return: parser
        """
        return self.parser.parse_args()
