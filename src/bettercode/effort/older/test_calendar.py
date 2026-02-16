import pytest
from calendar_tree import MaintenanceScheduler

def test_basic_non_overlap():
    scheduler = MaintenanceScheduler(100)
    assert scheduler.schedule(0, 10, "Job1") is None
    assert scheduler.schedule(20, 30, "Job2") is None
    # Verify Job1 is still there
    assert scheduler.schedule(5, 6, "Job3") == "Job1"

def test_basic_overlap():
    scheduler = MaintenanceScheduler(100)
    scheduler.schedule(10, 20, "Job1")
    # Overlap 15-25
    assert scheduler.schedule(15, 25, "Job2") == "Job1"

def test_straddle_bug():
    """
    THE KILLER TEST
    Range: 0 to 100. Midpoint: 50.
    
    1. Insert "LongJob" (40, 60).
       - Starts in Left (40 < 50). Ends in Right (60 > 50).
       - Logic: `if end <= mid` (60 <= 50) is False.
       - Goes to ELSE -> Right Child.
       - "LongJob" is stored in the subtree covering [50, 100].
       
    2. Insert "SmallJob" (42, 45).
       - Overlaps with LongJob (40-60).
       - Starts 42, Ends 45.
       - Logic: `find_overlap` sees start < mid. Recurses Left.
       - Left Child covers [0, 50].
       - Left Child is EMPTY (because LongJob went Right).
       - Returns None (No Conflict).
    """
    scheduler = MaintenanceScheduler(100)
    
    # 1. Schedule a job that crosses the midpoint (50)
    assert scheduler.schedule(40, 60, "LongJob") is None
    
    # 2. Schedule a small job entirely within the first half of LongJob
    # This SHOULD fail because 42-45 overlaps with 40-60.
    conflict = scheduler.schedule(42, 45, "SmallJob")
    
    assert conflict == "LongJob", \
        f"Logic Error: SmallJob (42-45) should collide with LongJob (40-60), but got {conflict}"

if __name__ == "__main__":
    # Manual run wrapper
    try:
        test_straddle_bug()
        print("Test Passed (Bug Fixed!)")
    except AssertionError as e:
        print(f"Test Failed (Bug Triggered): {e}")
