"""
    Evaluation:
        Evaluations of the models
"""
from sklearn.metrics import f1_score, accuracy_score, classification_report


def f1_macro(gold, predicts, average):
    return f1_score(gold, predicts, average=average)


def accuracy(gold, predicts):
    return accuracy_score(gold, predicts)


def classification_report_print(gold, predicts):
    report = classification_report(gold, predicts)
    return report


def evaluate(gold, predicts, average):
    f1 = f1_macro(gold, predicts, average=average)
    acc = accuracy(gold, predicts)
    report = classification_report(gold, predicts)
    return f1, acc, report
