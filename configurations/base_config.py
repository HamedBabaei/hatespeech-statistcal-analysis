"""
    BaseConfig
         Get model parameters
"""
import argparse
from pathlib import Path
import os


class BaseConfig:
    """ 
        Base Configigurations
    """
    def __init__(self):
        dataset = "twitter"
        dataset_config = {
            "twitter":{
                'loader': "csv",
                'train': "twitter_train.csv",
                'label': 'label',
                'test' : "twitter_test.csv",
            }
        }
        target_transformer = {
            "twitter": {"0": 0, "1":1},
        }
        self.parser = argparse.ArgumentParser()

        # dataset and dataset directory
        self.parser.add_argument("--dataset", type=str, default=dataset,
                                help='Dataset name')
        self.parser.add_argument("--intermediate_train_dir", type=str,
                                 default=os.path.join(Path(__file__).parents[1].__str__(),
                                                          f"datasets/{dataset}/intermediate"),
                                 help='Path to intermediate directory')
        self.parser.add_argument("--logs_dir", type=str,
                                 default=os.path.join(Path(__file__).parents[1].__str__(), "logs"),
                                 help='Path to logs directory')

        # train, dev, and test file name
        self.parser.add_argument("--train_name", type=str, default=dataset_config[dataset]['train'],
                                 help='Name of train file')
        self.parser.add_argument("--test_name", type=str, default=dataset_config[dataset]['test'],
                                 help='Name of test file')

        # dataset loader and class label
        self.parser.add_argument("--loader", type=str, default=dataset_config[dataset]['loader'],
                                 help='Loader of the dataset')
        self.parser.add_argument("--label", type=str, default=dataset_config[dataset]['label'],
                                 help='Class label for grand truth')
        
        # Representation config and target transformers
        self.parser.add_argument("--target_transformer", default=target_transformer[dataset],
                                 help='target transformer')
        self.parser.add_argument("--cuda", type=bool, default=True,
                                 help='Cuda Initializations')
        
        # General Model configurations
        self.parser.add_argument("--model_name", type=str, default="lr",
                                 help='Model name, LogisticRegression or BERT (lr, bert)')
        self.parser.add_argument("--test", type=bool, default=False,
                                 help='To do training or testing?')

        # TFIDF + LogisticRegression Configurations
        self.parser.add_argument("--ngram_range", type=tuple, default=(2, 3),
                                 help='N-gram model range')
        self.parser.add_argument("--sublinear_tf", type=bool, default=True,
                                 help='Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)')
        self.parser.add_argument("--stop_words", type=str, default=None,
                                 help='Stop words list, the default is None, means no stop word removings')
        self.parser.add_argument("--C", type=float, default=1,
                                 help='Inverse of regularization strength')
        

    def get_args(self):
        """
            Return parser
        :return: parser
        """
        return self.parser.parse_args()
