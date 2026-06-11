"""Routing accuracy evaluation."""


def compute_routing_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Compute the fraction of cases routed to the expected team.

    Args:
        y_true: List of expected (gold) team identifiers.
        y_pred: List of predicted team identifiers, same length as `y_true`.

    Returns:
        float: The routing accuracy as a value in `[0, 1]`.

    Raises:
        ValueError: If `y_true` and `y_pred` have different lengths or
            are empty.
    """
    if not y_true or not y_pred:
        raise ValueError("y_true and y_pred must be non-empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return correct / len(y_true)
