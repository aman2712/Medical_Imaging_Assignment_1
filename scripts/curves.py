import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def calculate_curves(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels,
        probabilities
    )
    precision, recall, _ = precision_recall_curve(
        labels,
        probabilities
    )

    return {
        "fpr": false_positive_rate,
        "tpr": true_positive_rate,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc(false_positive_rate, true_positive_rate),
        "pr_auc": auc(recall, precision),
    }
