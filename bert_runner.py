"""
    Hate Speech
"""
import torch
from configurations import BaseConfig
from models import roberta_twitter_train_model, roberta_twitter_test_model
from transformers import set_seed
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    CONFIG = BaseConfig().get_args()
    set_seed(CONFIG.seed_num)

    DEVICE = 'cpu'
    if CONFIG.cuda:
        DEVICE = ("cuda:1" if torch.cuda.is_available() else 'cpu')
    if not CONFIG.test:
        roberta_twitter_train_model(CONFIG, DEVICE)
        pass
    else:
        if CONFIG.dataset == 'twitter':
            roberta_twitter_test_model(CONFIG, DEVICE)
        else:
            pass
