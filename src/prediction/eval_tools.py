"""Helper functions for evaluating the performance of the classifier."""

from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

sns.set_theme()


def validate_by_target_cell(classifier: tf.keras.Model, val_inputs: list,
                            val_labels: np.ndarray) -> dict:
    """
    Calculates validation accuracy for individual target cells.

    Parameters
    ----------
    classifier : Model
        The classifier to be evaluated.
    val_inputs : list
        A list formatted according to how the classifier expects input. Targets are located in last
        position of the list.
    val_labels : np.ndarray
        The ground truth labels for the provided inputs.

    Returns
    -------
        A dictionary containing each target cell as a key and the performance metrics for the given
        cell - structured as another dict - as that key's value. The metrics dictionary also
        contains label imbalance for the given cell in binary classification tasks.
    """
    # Determine target cell index from one-hot encoded array.
    target_cell_indicators = np.where(val_inputs[-1] == 1)[1]

    metrics = {}
    for i in set(target_cell_indicators):
        relevant_data_indices = np.where(target_cell_indicators == i)
        cell_inputs = []
        for input_array in val_inputs:
            cell_inputs.append(input_array[relevant_data_indices])
        cell_labels = val_labels[relevant_data_indices]
        cell_key = f"cell_{i}"
        metrics[cell_key] = classifier.test_on_batch(cell_inputs,
                                                     cell_labels,
                                                     return_dict=True)
        if val_labels.shape[1] == 2:
            metrics[cell_key][
                "label_balance"] = determine_binary_label_balance(cell_labels)
    return metrics


def determine_binary_label_balance(labels: np.ndarray) -> float:
    """
    Calculates the distribution of labels in a given array. The resulting value will be somewhere
    between 0 (all labels have the same value) or 1 (perfect balance between labels).
    """
    counter = Counter(np.where(labels == 1)[1])
    # If there is not perfect balance between the two labels, the smaller number of occurrences has
    # to be the numerator to make sure we only get values <= 1.
    if counter[0] < counter[1]:
        return counter[0] / counter[1]
    return counter[1] / counter[0]


def plot_label_imbalance_against_accuracy(metrics: dict):
    """
    Creates a lineplot of accuracy and label balance per cell.
    """
    accuracies = []
    balances = []
    for cell_metric in metrics.values():
        accuracies.append(cell_metric["accuracy"])
        balances.append(cell_metric["label_balance"])
    data = pd.DataFrame.from_dict({
        "accuracy": accuracies,
        "balance": balances
    })
    sns.scatterplot(data=data, x="balance", y="accuracy")


def calculate_metrics(predictions: np.ndarray,
                      ground_truth: np.ndarray,
                      mode: str = "binary",
                      return_dict: bool = False):
    """
    Entrypoint for metrics functions.
    """
    mode = mode.lower()
    if mode == "binary":
        metrics = _calculate_binary_metrics(predictions, ground_truth)
    elif mode == "real-value":
        metrics = _calculate_real_value_metrics(predictions, ground_truth)
    else:
        raise ValueError(
            "mode parameter value needs to be either 'binary' or 'real-value'."
        )
    if return_dict:
        return metrics


def _calculate_binary_metrics(predictions: np.ndarray,
                              ground_truth: np.ndarray):
    """
    Calculate all sorts of metrics for a binary classification task.
    """
    # Transform one_hot encoded labels into integers.
    predictions = np.argmax(predictions, axis=1)
    ground_truth = np.argmax(ground_truth, axis=1)
    con_matrix = confusion_matrix(y_true=ground_truth, y_pred=predictions)
    true_negatives, false_positives, false_negatives, true_positives = con_matrix.ravel(
    )
    positives = true_positives + false_negatives
    negatives = true_negatives + false_positives
    fpr = false_positives / negatives
    print(f"False Positive Rate: {fpr}")
    fnr = false_negatives / positives
    print(f"False Negatives Rate: {fnr}")
    precision = precision_score(y_true=ground_truth, y_pred=predictions)
    print(f"Precision: {precision}")
    recall = recall_score(y_true=ground_truth, y_pred=predictions)
    print(f"Recall: {recall}")
    f1 = f1_score(y_true=ground_truth, y_pred=predictions)
    print(f"f1: {f1}")
    return {
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _calculate_real_value_metrics(predictions: np.ndarray,
                                  ground_truth: np.ndarray) -> dict:
    """
    Calculate all sorts of metrics for a real-value classification task.
    """
    # Get indices of positives.
    positives = np.where(ground_truth > 0)[0]
    true_positives = np.where(predictions[positives] > 0)[0]
    grid_hit_rate = len(true_positives) / len(positives)
    print(f"Grid Hit Rate: {grid_hit_rate}")

    total_occurrences = ground_truth[positives].sum()
    total_predicted_occurrences = ground_truth[true_positives].sum()
    case_hit_rate = total_predicted_occurrences / total_occurrences
    print(f"Total occurrences: {total_occurrences}")
    print(f"Total predicted occurrences: {total_predicted_occurrences}")
    print(f"Case Hit Rate: {case_hit_rate}")

    hit_efficiency_index = case_hit_rate / grid_hit_rate
    print(f"Hit Efficiency Index: {hit_efficiency_index}")
    return {"case_hit_rate": case_hit_rate, "hit_efficiency_index": hit_efficiency_index}
