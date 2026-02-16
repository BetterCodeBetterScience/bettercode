import pytest
from median import RollingMedian

def test_basic_median():
    rm = RollingMedian()
    for x in [1, 2, 3]:
        rm.add_latency(x)
    # 1, 2, 3 -> Median 2
    assert rm.get_median() == 2

def test_ghost_balance_failure():
    """
    THE KILLER TEST.
    
    1. Insert [10, 10, 10, 20, 20, 20].
       lo = [10, 10, 10] (len 3)
       hi = [20, 20, 20] (len 3)
       Median = 15.
       
    2. Remove all 10s.
       Lazy delete marks 10 as deleted.
       Heaps are UNTOUCHED physically.
       lo = [10, 10, 10] (len 3, all ghosts)
       hi = [20, 20, 20] (len 3, all real)
       
    3. Call get_median().
       _clean_top() runs on 'lo'.
       It pops the first 10. 'lo' is now [10, 10] (len 2).
       Wait... the lazy clean pops them ALL because they are at the top.
       'lo' becomes EMPTY.
       
       State after clean:
       lo = [] (len 0)
       hi = [20, 20, 20] (len 3)
       
       Logic: len(hi) > len(lo). Returns hi[0] -> 20.
       Correct Median of [20, 20, 20] is 20.
       
       Wait, this case passes by accident! 
       We need to bury the ghosts so _clean_top() doesn't remove them all.
    """
    
    rm = RollingMedian()
    
    # 1. Construct a state where ghosts are buried under a valid entry
    # Insert 5 (lo), 25 (hi)
    # Insert 10 (lo), 20 (hi)
    # Insert 15 (lo), 15 (hi) - Rebalance might swap
    
    # Simple setup:
    # Add: 5, 10, 20, 25.
    # lo=[10, 5], hi=[20, 25]. (Balanced)
    
    rm.add_latency(5)
    rm.add_latency(20)
    rm.add_latency(10)
    rm.add_latency(25)
    
    # Check sanity
    assert rm.get_median() == 15.0 # (10+20)/2
    
    # 2. Delete '5'.
    # 5 is in 'lo', but it is NOT at the top (10 is at the top).
    rm.remove_latency(5)
    
    # State:
    # lo = [10, 5(ghost)]. Len = 2.
    # hi = [20, 25].       Len = 2.
    # Real Left: [10]. Real Right: [20, 25].
    # Total Real: 10, 20, 25.
    # True Median: 20.
    
    # Code Execution:
    # _clean_top(): lo top is 10. Not deleted. Does nothing.
    # Checks lengths: len(lo) == 2, len(hi) == 2.
    # Returns average(lo[0], hi[0]) -> (10 + 20) / 2 = 15.0.
    
    # ERROR: Returns 15.0. Correct is 20.
    
    val = rm.get_median()
    assert val == 20, f"Failed Hidden Ghost Test. Got {val}, expected 20. Ghosts skewed the balance check."

def test_mass_deletion_skew():
    """
    A more dramatic skew.
    Data: [1..100]. Median 50.5.
    Remove [1..49].
    Remaining: [50..100]. (51 items).
    Median should be 75.
    
    Buggy code will think the lower heap is still fully populated with ghosts
    and likely return something near 50.
    """
    rm = RollingMedian()
    for i in range(1, 101):
        rm.add_latency(i)
        
    # Remove lower half (except 50)
    for i in range(1, 50):
        rm.remove_latency(i)
        
    # lo heap has 1..50 (49 ghosts). len = 50.
    # hi heap has 51..100. len = 50.
    # clean_top checks lo[0] -> 50. It is NOT deleted.
    # Code sees len 50 vs len 50.
    # Returns avg(50, 51) = 50.5.
    
    # Valid set: 50, 51, ... 100.
    # Median of 50..100 is 75.
    
    result = rm.get_median()
    assert result == 75, f"Failed Mass Deletion. Got {result}, expected 75."

if __name__ == "__main__":
    test_ghost_balance_failure()
    test_mass_deletion_skew()
    print("Tests passed (Algorithm is correct!)")
