import pytest
from intervals import Interval, merge_overlapping


def test_does_not_merge_touching_intervals():
    # Touching at point 2 should NOT merge per spec.
    intervals = [Interval(1, 2), Interval(2, 3)]
    assert merge_overlapping(intervals) == [Interval(1, 2), Interval(2, 3)]


def test_merges_actual_overlaps():
    intervals = [Interval(1, 3), Interval(2, 4)]
    assert merge_overlapping(intervals) == [Interval(1, 4)]


def test_nested_interval():
    intervals = [Interval(1, 10), Interval(3, 5)]
    assert merge_overlapping(intervals) == [Interval(1, 10)]


def test_multiple_with_gaps_and_touches():
    intervals = [
        Interval(1, 2),
        Interval(2, 3),   # touches [1,2] -> should NOT merge
        Interval(10, 12),
        Interval(11, 15), # overlaps -> should merge into [10,15]
        Interval(15, 18), # touches [10,15] -> should NOT merge
    ]
    assert merge_overlapping(intervals) == [
        Interval(1, 2),
        Interval(2, 3),
        Interval(10, 15),
        Interval(15, 18),
    ]


def test_empty_input():
    assert merge_overlapping([]) == []


def test_invalid_interval_raises():
    with pytest.raises(ValueError):
        Interval(5, 4)
