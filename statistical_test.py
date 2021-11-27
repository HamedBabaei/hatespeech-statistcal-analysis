from configurations import TestConfig
from datahandlers import DataReader, DataWriter

if __name__=='__main__':
    CONFIG = TestConfig().get_args()
    ML_EVAL = DataReader.load_json(CONFIG.ml_clf)
    RoBERTa_EVAL = DataReader.load_json(CONFIG.roberta_clf)

    assert ML_EVAL['gt'] == RoBERTa_EVAL['gt']
    assert ML_EVAL['predict'] != RoBERTa_EVAL['predict']
    assert len(ML_EVAL['predict']) == len(RoBERTa_EVAL['predict'])



