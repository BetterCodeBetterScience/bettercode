from bettercode.bug_driven_testing import find_outliers
import pytest

def test_find_outliers_normal_case():
    data = [1, 2, 3, 4, 5, 100]  # 100 is clearly an outlier
    outliers = find_outliers(data, threshold=2.0)
    
    # Should find the outlier at index 5
    # Should find the outlier at index 5
    assert 5 in outliers, f"Failed to detect outlier: {outliers}"
    assert len(outliers) == 1, f'Expected exactly one outlier, got: {len(outliers)}'

def test_find_outliers_identical_values():
    data = [5, 5, 5, 5, 5]  # All identical values
    
    outliers = find_outliers(data, threshold=2.0)
    assert outliers == [], f"Expected no outliers for identical values, got {outliers}"


# def test_find_outliers_edge_cases():
#     """Test find_outliers with edge cases."""
#     # Empty list
#     assert find_outliers([]) == []
    
#     # Single element
#     assert find_outliers([5]) == []
    
#     # Two identical elements
#     try:
#         outliers = find_outliers([5, 5])
#         assert outliers == []
#     except ZeroDivisionError:
#         assert False, "Function crashed with two identical values"
