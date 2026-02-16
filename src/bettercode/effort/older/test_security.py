import pytest
from security import PerimeterMonitor

def test_basic_linear_sector():
    """
    Standard case: Cameras at 10 and 50.
    Camera 50 covers (10, 50].
    Event at 30 -> Should be True.
    """
    pm = PerimeterMonitor()
    pm.add_camera(10)
    pm.add_camera(50)
    
    assert pm.is_responsible(50, 30) is True
    assert pm.is_responsible(50, 10) is False # Boundary belongs to Prev
    assert pm.is_responsible(50, 50) is True  # Boundary belongs to Curr

def test_midnight_wrap_around():
    """
    THE KILLER TEST.
    
    Setup:
    - Camera A at 300 degrees.
    - Camera B at 45 degrees.
    
    Topologically:
    - Camera A covers (45, 300].
    - Camera B covers (300, 360) U [0, 45].
    
    Scenario:
    - Event happens at 20 degrees.
    - Visually, 20 is between 300 and 45 (going clockwise).
    - Camera B should be responsible.
    
    Execution Trace:
    - is_responsible(camera=45, event=20)
    - index of 45 is 0 (since 45 < 300, list is [45, 300]).
    - sector_start = cameras[-1] -> 300.
    - sector_end = 45.
    - Check: if 300 < 20 <= 45:
    - Result: False.
    
    Bug: The event is rejected because the code doesn't handle the split interval.
    """
    pm = PerimeterMonitor()
    pm.add_camera(300)
    pm.add_camera(45)
    
    # List sorts to: [45, 300]
    
    # Case 1: Event at 20 (Past 0, before 45).
    # Responsible camera should be 45.
    # Previous camera is 300.
    # Sector: (300 -> 360 -> 45).
    assert pm.is_responsible(45, 20) is True, \
        "Failed Wrap-Around: Event at 20 should be covered by Camera 45 (Start 300)"

    # Case 2: Event at 350 (Past 300, before 360).
    # Responsible camera should be 45.
    assert pm.is_responsible(45, 350) is True, \
        "Failed Wrap-Around: Event at 350 should be covered by Camera 45 (Start 300)"

if __name__ == "__main__":
    try:
        test_midnight_wrap_around()
        print("Test Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
