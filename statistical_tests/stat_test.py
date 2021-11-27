import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from pprint import pprint
from sklearn.metrics import confusion_matrix, cohen_kappa_score, matthews_corrcoef

def McNemar(gold, clf1_preds, clf2_preds):
    """
        McNemar Test
                    	Test 2 positive	Test 2 negative	    total
        Test 1 positive	        a	            b	        a + b
        Test 1 negative     	c	            d	        c + d
        Column total	      a + c	          b + d	          N

    """
    clf1_pred_eval = [1 if gt == clf1_preds[i] else 0 for i, gt in enumerate(gold)]
    clf2_pred_eval = [1 if gt == clf2_preds[i] else 0 for i, gt in enumerate(gold)]
    a, b, c, d = 0, 0, 0, 0
    for i, _ in enumerate(clf1_pred_eval):
        if clf1_pred_eval[i] == clf2_pred_eval[i]:
            if clf1_pred_eval[i] == 0:
                d += 1
            else:
                a += 1
        else:
            if clf1_pred_eval[i] == 0:
                c += 1
            else:
                b += 1
    contingency_table = np.array([[a, b],[c, d]])
    pprint(contingency_table)
    result = mcnemar(contingency_table, exact=False, correction=True)
    print(f'statistic={result.statistic}, p-value={result.pvalue}')
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')


def confidence_interval(skill, n):
    print("Confidence Interval Test")
    CI = {"90%":1.64, "95%":1.96, "98%":2.33, "99%":2.58}
    F = lambda error, const, n : const * np.sqrt((error * (1 - error)) / n)
    for ci, const in CI.items():
        print(f"* {ci} likelihood that the CI {[0, F(1-skill, const, n)]} "
              f"covers the true classification error of the model on unseen data")

def errors(gold, predict):
    FP = confusion_matrix(gold, predict)[0][1]  # Type I Error
    TN = confusion_matrix(gold, predict)[1][1]
    FPR = FP/(FP + TN)                          # FPR or Fall-out
    FN = confusion_matrix(gold, predict)[1][0]  # Type II Error
    TP = confusion_matrix(gold, predict)[0][0]
    FNR = FN/(FN + TP)                          # FNR or Miss rate
    print(f"Type  I Error(FP) is : {FP},  FPR (Fall-out) : {FPR}")
    print(f"Type II Error(FN) is : {FN},  FNR (Miss rate): {FNR}")

def kappa(gold, predict):
    """
    Interrater reliability: the kappa statistic
    .40–.59	    Weak	        15–35%
    .60–.79	    Moderate	    35–63%
    .80–.90	    Strong	        64–81%
    Above.90    Almost Perfect	82–100%
    """
    print(f"Cohen Kappa Score:{cohen_kappa_score(gold, predict)}")

def phi_coefficient(gold, predict):
    print(f"Compute the Matthews correlation coefficient (MCC):{matthews_corrcoef(gold, predict)}")
