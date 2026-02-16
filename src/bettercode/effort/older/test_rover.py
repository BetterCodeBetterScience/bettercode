import pytest
from rover import solve_rover_path

def test_choke_point_trap():


    grid = [
        [0,   0,   0,   0],    # Row 0: S at (0,0), Target at (0,3)
        [100, 100, 100, 100],  # Row 1: Wall
        [-1,  100, 100, 100]   # Row 2: Charger at (2,0)
    ]
        
    grid = [
        [0,   0,   0,   0],    # (0,2) is the Choke. (0,3) is Target.
        [100, 100, 50,  100],  # (1,2) allows passage up
        [-1,  50,  50,  100]   # (2,0) -> (2,1) -> (2,2)
    ]
        
    grid_final = [
        [0,   1,   1,   0,   0],    # Row 0. Target (0,4)
        [-1,  100, 100, 100, 100]   # Row 1. Charger at (1,0)
    ]
    # Note: `solve_rover_path` assumes square grid usually? 
    # The code says `n = len(grid)`, `target = (n-1, n-1)`.
    # We must use a square grid.
    
    grid_square = [
        [0,   1,   1,   0,   0], # (0,3) is Choke. (0,4) Target? No, T is (4,4)
        [-1,  100, 100, 100, 100],
        [100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100],
        [100, 100, 100, 100, 0]  # T is (4,4)
    ]

    grid_3x3 = [
        [0,   0,   0],    # A: (0,0)->(0,1)->(0,2)
        [-1,  100, 0],    # B: (0,0)->(1,0)[C]. Merge at (1,2).
        [100, 100, 0]     # Target (2,2)
    ]
    
    max_battery = 3
    result = solve_rover_path(grid_3x3, max_battery)
    
    if result == -1:
        pytest.fail("Logic Trap Triggered: The algorithm pruned the high-cost (charged) path "
                    "because a low-cost (drained) path blocked the node.")
    else:
        # If it returns a value (likely 100), the bug is fixed/not triggered.
        pass

if __name__ == "__main__":
    test_choke_point_trap()
