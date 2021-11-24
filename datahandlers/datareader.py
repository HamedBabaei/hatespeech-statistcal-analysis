"""
Data Readers for loading various datasets
"""
import pandas as pd
from pathlib import Path
import json
import pickle
import codecs


class DataReader:
    """
        Data loader class for loading datas and files
    """
    @staticmethod
    def load_pkl(path: Path) -> pickle:
        pass

    @staticmethod
    def load_json(path: Path) -> json:
        """
            loading a json file
        :param path:
        :return:
        """
        with codecs.open(path, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
        return json_data

    @staticmethod
    def load_csv(path: Path) -> pd:
        """
            loading a csv file
        :param path:
        :return:
        """
        df = pd.read_csv(path)
        return df

    @staticmethod
    def load_excel(path: Path) -> pd:
        """
            loading excel file
        :param path:
        :return:
        """
        excel = pd.read_excel(path)
        return excel
        
    @staticmethod
    def load_text(path: Path) -> str:
        """
            loading text file
        :param path:
        :return:
        """
        with codecs.open(path, 'r', encoding='utf-8') as myfile:
            text = myfile.read()
        return text
