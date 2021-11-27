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
                'loader': "load_csv",
                'train' : "twitter_train.csv",
                'label' : 'label',
                "val"   : "twitter_val.csv",
                'test'  : "twitter_test.csv",
            }
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
        self.parser.add_argument("--val_name", type=str, default=dataset_config[dataset]['val'],
                                 help='Name of validation file')
        self.parser.add_argument("--test_name", type=str, default=dataset_config[dataset]['test'],
                                 help='Name of test file')
        # dataset loader and class label
        self.parser.add_argument("--loader", type=str, default=dataset_config[dataset]['loader'],
                                 help='Loader of the dataset')
        
        # General Model configurations
        self.parser.add_argument("--model_name", type=str, default="roberta",
                                 help='Model name, ML(SVM) or BERT (ml, roberta)')
        self.parser.add_argument("--test", type=bool, default=False,
                                 help='To do training or testing?')
        self.parser.add_argument("--pre_trained_dir", type=str,
                                 default=os.path.join(Path(__file__).parents[1].__str__(), "pretrained-model"),
                                 help='path to the pretrained model directory?')
        self.parser.add_argument("--seed_num", type=int, default=222,
                                 help="Default Seed Num to regenerate results")

        # TFIDF + ML Configurations
        self.parser.add_argument("--ngram_range", type=tuple, default=(2, 4),
                                 help='N-gram model range')
        self.parser.add_argument("--sublinear_tf", type=bool, default=True,
                                 help='Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)')
        self.parser.add_argument("--stop_words", type=str, default=None,#"english",
                                 help='Stop words list, the default is None, means no stop word removings')
        self.parser.add_argument("--analyzer", type=str, default="char",
                                 help='Character Ngram Model')

        # RoBERTa
        self.parser.add_argument("--cuda", type=bool, default=True,
                                 help='Cuda Initializations')
        self.parser.add_argument("--roberta_base", type=str,
                                 default="/mnt/disk2/transformers/roberta",
                                 help='Path to roberta pre-trained model')
        self.parser.add_argument("--roberta_fine_tune", type=str,
                                 default="/mnt/disk2/hbmodels/hatespeech-roberta",
                                 help='Path to fine-tuned version of roberta for the task')
        self.parser.add_argument("--checkpoint", type=str, default="checkpoint-1000",
                                 help='Fine-tuned checkpoint')
        self.parser.add_argument("--max_length", type=int, default=512, help='maximum length')
        self.parser.add_argument("--save_total_limit", type=int, default=1, help='Save limits')
        self.parser.add_argument("--batch_size", type=int, default=8, help='Batch Size')
        self.parser.add_argument("--epoch", type=int, default=3, help='Epoch Number')
        self.parser.add_argument("--weight_decay", type=float, default=0.0001, help='Weight Decay')

    def get_args(self):
        """
            Return parser
        :return: parser
        """
        return self.parser.parse_args()
