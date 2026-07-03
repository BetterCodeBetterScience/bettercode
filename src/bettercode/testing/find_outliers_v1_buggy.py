def find_outliers(data: list[float], threshold: float = 2.0) -> list[int]:
    """Find outliers in a dataset using the z-score method.

    Args:
        data: Numerical values to scan.
        threshold: Number of standard deviations from the mean above which a
            value is flagged as an outlier.

    Returns:
        Indices of the outlying values in ``data``.
    """
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = variance ** 0.5

    # Bug: division by zero when std is 0 (all values are identical)
    # This only happens when all data points are the same
    outliers = []
    for i, value in enumerate(data):
        z_score = abs(value - mean) / std
        if z_score > threshold:
            outliers.append(i)

    return outliers
