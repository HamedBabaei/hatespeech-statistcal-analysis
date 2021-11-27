from configurations import TestConfig
from datahandlers import DataReader, DataWriter
from statistical_tests import McNemar, confidence_interval, errors, kappa, phi_coefficient

if __name__=='__main__':
    CONFIG = TestConfig().get_args()
    ML_EVAL = DataReader.load_json(CONFIG.ml_clf)
    RoBERTa_EVAL = DataReader.load_json(CONFIG.roberta_clf)

    assert ML_EVAL['gt'] == RoBERTa_EVAL['gt']
    assert ML_EVAL['predict'] != RoBERTa_EVAL['predict']
    assert len(ML_EVAL['predict']) == len(RoBERTa_EVAL['predict'])

    print("-----------------------McNemar----------------------------")
    GT, ML_PRED, RoBERTa_PRED = RoBERTa_EVAL['gt'], ML_EVAL['predict'], RoBERTa_EVAL['predict']
    McNemar(gold=GT, clf1_preds=ML_PRED, clf2_preds=RoBERTa_PRED)

    print("--------------------------CI------------------------------")
    print("ML MODEL")
    confidence_interval(skill=ML_EVAL['F1 Macro'], n=len(ML_EVAL['gt']))
    print("\nRoBERTa")
    confidence_interval(skill=RoBERTa_EVAL['F1 Macro'], n=len(RoBERTa_EVAL['gt']))
    print("-------------------ERROR TYPE I, II------------------------")
    print("ML MODEL")
    errors(gold=ML_EVAL['gt'], predict=ML_EVAL['predict'])
    print("\nRoBERTa")
    errors(gold=RoBERTa_EVAL['gt'], predict=RoBERTa_EVAL['predict'])

    print("------------------------Kappa Score------------------------")
    print("ML MODEL")
    kappa(gold=ML_EVAL['gt'], predict=ML_EVAL['predict'])
    print("\nRoBERTa")
    kappa(gold=RoBERTa_EVAL['gt'], predict=RoBERTa_EVAL['predict'])
    print("\nML MODEL + RoBERTa")
    kappa(gold=ML_EVAL['predict'], predict=RoBERTa_EVAL['predict'])
    print("------------------------Phi Coefficient----------------------")
    print("ML MODEL")
    phi_coefficient(gold=ML_EVAL['gt'], predict=ML_EVAL['predict'])
    print("\nRoBERTa")
    phi_coefficient(gold=RoBERTa_EVAL['gt'], predict=RoBERTa_EVAL['predict'])


