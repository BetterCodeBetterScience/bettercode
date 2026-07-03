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

    if std == 0:
        # If standard deviation is zero, all values are identical,
        # so no outliers
        return []

    outliers = []
    for i, value in enumerate(data):
        z_score = abs(value - mean) / std
        if z_score > threshold:
            outliers.append(i)

    return outliers
