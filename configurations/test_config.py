"""
    TestConfig
         Get test parameter for statistical testings
"""
import argparse
from pathlib import Path
import os


class TestConfig:
    """
        Test Configurations
    """
    def __init__(self):
        """
            Test Configurations
        """
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--ml_clf", type=str,
                                 default=os.path.join(Path(__file__).parents[1].__str__(),
                                                      "logs", "twitter-evaluation-ml.json"),
                                 help='Path to ml classifier directory')
        self.parser.add_argument("--roberta_clf", type=str,
                                 default=os.path.join(Path(__file__).parents[1].__str__(),
                                                      "logs", "twitter-evaluation-roberta.json"),
                                 help='Path to intermediate directory')
        self.parser.add_argument("--logs_dir", type=str,
                                 default=os.path.join(Path(__file__).parents[1].__str__(), "logs"),
                                 help='Path to logs directory to save the stats')

    def get_args(self):
        """
            Return parser
        :return: parser
        """
        return self.parser.parse_args()
