from configurations import TestConfig
from datahandlers import DataReader, DataWriter
from statistical_tests import McNemar, confidence_interval, errors, kappa, phi_coefficient

if __name__=='__main__':
    CONFIG = TestConfig().get_args()
    ML_EVAL = DataReader.load_json(CONFIG.ml_clf)
    RoBERTa_EVAL = DataReader.load_json(CONFIG.roberta_clf)

    assert ML_EVAL['Test-gt'] == RoBERTa_EVAL['Test-gt']
    assert ML_EVAL['Test-predict'] != RoBERTa_EVAL['Test-predict']
    assert len(ML_EVAL['Test-predict']) == len(RoBERTa_EVAL['Test-predict'])

    print("-----------------------McNemar----------------------------")
    GT, ML_PRED, RoBERTa_PRED = RoBERTa_EVAL['Test-gt'], ML_EVAL['Test-predict'], RoBERTa_EVAL['Test-predict']
    McNemar(gold=GT, clf1_preds=ML_PRED, clf2_preds=RoBERTa_PRED)

    print("--------------------------CI------------------------------")
    print("ML MODEL")
    confidence_interval(gold=ML_EVAL['Test-gt'], predict=ML_EVAL['Test-predict'])
    print("\nRoBERTa")
    confidence_interval(gold=RoBERTa_EVAL['Test-gt'], predict=RoBERTa_EVAL['Test-predict'])
    print("-------------------ERROR TYPE I, II------------------------")
    print("ML MODEL")
    errors(gold=ML_EVAL['Test-gt'], predict=ML_EVAL['Test-predict'])
    print("\nRoBERTa")
    errors(gold=RoBERTa_EVAL['Test-gt'], predict=RoBERTa_EVAL['Test-predict'])

    print("------------------------Kappa Score------------------------")
    print("ML MODEL")
    kappa(gold=ML_EVAL['Test-gt'], predict=ML_EVAL['Test-predict'])
    print("\nRoBERTa")
    kappa(gold=RoBERTa_EVAL['Test-gt'], predict=RoBERTa_EVAL['Test-predict'])
    print("\nML MODEL + RoBERTa")
    kappa(gold=ML_EVAL['Test-predict'], predict=RoBERTa_EVAL['Test-predict'])
    print("------------------------Phi Coefficient----------------------")
    print("ML MODEL")
    phi_coefficient(gold=ML_EVAL['Test-gt'], predict=ML_EVAL['Test-predict'])
    print("\nRoBERTa")
    phi_coefficient(gold=RoBERTa_EVAL['Test-gt'], predict=RoBERTa_EVAL['Test-predict'])


