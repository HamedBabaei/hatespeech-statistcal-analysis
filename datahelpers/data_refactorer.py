"""
Creates processed dataset for classification procedure
"""
import os
from datahandlers import DataReader, DataWriter
from .twitter_refactorer import twitter

class DataRefactorer:

    def __init__(self, config, SPAN="::[DATA REFACTOR]::",
                 datasets=["twitter", "pan2021"]):
        print(f"[START] {SPAN}")
        self.datasets = datasets
        _ = self.dataset_validator(config.dataset)
        self.make_dir(config.intermediate_train_dir)
        eval(config.dataset)(SPAN=SPAN + config.dataset.upper() + "::",
                             config=config,
                             data_reader=DataReader,
                             data_writer=DataWriter)
        print("[END] Refactoring")

    def dataset_validator(self, dataset):
        """
            validate dataset name
        :param dataset:
        :return:
        """
        if dataset in self.datasets:
            return True
        else:
            raise Exception(f"You should use the following datasets:{self.datasets}!")

    def make_dir(self, path: str):
        """
            check whatever directory is exist or not,
                if yes so it is ok
                else so create one
        :param path:
        :return:
        """
        if not os.path.exists(path):
            os.mkdir(path)
