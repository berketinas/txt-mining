import pandas as pd
from sklearn.metrics import roc_curve, auc, mean_squared_error
from math import sqrt
from IPython.display import display
from matplotlib import pyplot as plt


# precision = true positive / (true positive + false positive)
# recall = true positive / (true positive + false negative)
def precision_recall(true_pos, false_pos, false_neg):
    return true_pos / (true_pos + false_pos), true_pos / (true_pos + false_neg)


def f1_score(true_pos, false_pos, false_neg):
    precision, recall = precision_recall(true_pos, false_pos, false_neg)
    return 2 * ((precision * recall) / (precision + recall))


def confusion_matrix(actual, predicted):
    display(pd.crosstab(actual, predicted))


def roc(actual, predicted):
    fpr, tpr, threshold = roc_curve(actual, predicted)

    print('Area under ROC curve for validation set: ', auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label='Validation set AUC')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    ax.legend(loc='best')
    plt.show()


def rmse(actual, predicted):
    print('ROOT MEAN SQUARE ERROR: ', sqrt(mean_squared_error(actual, predicted)))
