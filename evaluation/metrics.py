"""Classification metrics for intent prediction evaluation."""

from sklearn.metrics import precision_recall_fscore_support


def compute_intent_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Compute precision, recall, and F1 for intent classification.

    Args:
        y_true: List of expected (gold) intent labels.
        y_pred: List of predicted intent labels, same length as `y_true`.

    Returns:
        dict[str, float]: A dictionary with macro-averaged `precision`,
            `recall`, and `f1` keys, plus `accuracy`.

    Raises:
        ValueError: If `y_true` and `y_pred` have different lengths or
            are empty.
    """
    if not y_true or not y_pred:
        raise ValueError("y_true and y_pred must be non-empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / len(y_true)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }
