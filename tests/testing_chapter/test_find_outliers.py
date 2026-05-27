from bettercode.testing.find_outliers_v2_fixed import find_outliers


def test_find_outliers_normal_case():
    data = [1, 2, 3, 4, 5, 100]  # 100 is clearly an outlier
    outliers = find_outliers(data, threshold=2.0)

    # Should find the outlier at index 5
    assert 5 in outliers, f"Failed to detect outlier: {outliers}"
    assert len(outliers) == 1, f'Expected exactly one outlier, got: {len(outliers)}'


def test_find_outliers_identical_values():
    data = [5, 5, 5, 5, 5]  # All identical values

    outliers = find_outliers(data, threshold=2.0)
    assert outliers == [], f"Expected no outliers for identical values, got {outliers}"
