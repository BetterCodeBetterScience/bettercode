import pytest
from resolver import DependencyResolver

def test_bi_directional_cycle_bug():
    """
    Test a simple cycle: Node 0 <--> Node 1.
    
    1. calculate(0):
       - Visits 0.
       - Recurses to 1.
       - 1 sees 0 is visited. Returns 0 (path ends).
       - IMPORTANT: The algorithm caches result for 1 as '0'.
       - 0 returns 1 + 0 = 1.
       
    2. calculate(1):
       - Visits 1.
       - Should recurse to 0 (which is unvisited in this context).
       - BUT, it hits the cache for 1.
       - Cache says '0'.
       - Returns 0.
       
    Correct Answer for calculate(1) is 1 (Path 1->0).
    """
    resolver = DependencyResolver()
    
    # 0 <--> 1
    resolver.add_connection(0, 1)
    resolver.add_connection(1, 0)
    
    # Step 1: Run from 0
    res0 = resolver.find_longest_chain(0)
    assert res0 == 1, f"First run failed. Expected 1, got {res0}"
    
    # Step 2: Run from 1
    # The cache from Step 1 ('1' is a leaf) will poison this run
    # where '1' is the root.
    res1 = resolver.find_longest_chain(1)
    
    assert res1 == 1, (
        f"Logic Error: Expected 1 (path 1->0), but got {res1}. "
        "The memoization cache persisted invalid context from the previous run."
    )

if __name__ == "__main__":
    try:
        test_bi_directional_cycle_bug()
        print("Test Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
