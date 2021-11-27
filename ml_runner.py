"""
    Hate Speech
"""
import os
from configurations import BaseConfig
from models import NgramML
from models import ml_twitter_train_model, ml_twitter_test_model
from datahandlers import DataReader
import random
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    CONFIG = BaseConfig().get_args()
    random.seed(CONFIG.seed_num)

    if not CONFIG.test:
        MODEL = NgramML(CONFIG)
        ml_twitter_train_model(MODEL, CONFIG)
    else:
        MODEL = DataReader.load_pkl(os.path.join(CONFIG.pre_trained_dir, CONFIG.model_name, CONFIG.model_name+".sav"))
        if CONFIG.dataset == 'twitter':
            ml_twitter_test_model(MODEL, CONFIG)
        else:
            pass

