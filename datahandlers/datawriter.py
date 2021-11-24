"""
Data Saver for saving various files
"""
import pandas as pd
from pathlib import Path
import json
import codecs
import pickle
import os


class DataWriter:
    """
        Father class for definition of other data loaders
    """
    @staticmethod
    def write_pkl(data, path: Path):
        pass

    @staticmethod
    def write_json(data: json, path: Path):
        """
            Write json file
        :param data: json data
        :param path: to to save json file
        :return:
        """
        with codecs.open(path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)

    @staticmethod
    def write_csv(data: pd, path: Path):
        """
            Write CSV file
        :param data: csv data
        :param path: to save pandas file
        :return:
        """
        df = pd.DataFrame(data=data)
        df.to_csv(path, index=False)

    @staticmethod
    def write_excel(data: pd, path: Path):
        """
              Write Excel file
          :param data: excel data
          :param path: to save pandas file
          :return:
        """
        data.to_csv(path, index=False)

