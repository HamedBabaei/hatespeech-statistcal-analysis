"""
    Hate Speech
"""
from configurations import BaseConfig
from models import NLR
from datahandlers import DataReader, DataWriter

if __name__ == '__main__':
    CONFIG = BaseConfig().get_args()
    MODEL = NLR(CONFIG)
    
