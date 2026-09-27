import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(labels, predictions, probabilities=None):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    confusion = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = confusion.ravel()

    metrics = {
        "confusion_matrix": confusion,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "accuracy": accuracy_score(labels, predictions),
        "sensitivity": recall_score(labels, predictions, zero_division=0),
        "specificity": (
            true_negative / (true_negative + false_positive)
            if (true_negative + false_positive) > 0
            else 0.0
        ),
        "precision": precision_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }

    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
        metrics["pr_auc"] = average_precision_score(labels, probabilities)

    return metrics
