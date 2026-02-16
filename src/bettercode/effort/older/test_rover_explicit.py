import pytest
from rover import solve_rover_path

def test_choke_point_trap():
    """
    Grid Layout (3x4):
    S = Start (0,0)
    T = Target (0,3) (Top Right)
    C = Charger (2,0)
    X = Choke Point (0,2) -- MUST go through here to reach T.
    
    Grid:
    [0,  0,  0,  0]  <-- Row 0: Path A (S -> X -> T)
    [10, 10, 10, 10] <-- Row 1: Wall
    [-1, 10, 10, 10] <-- Row 2: Charger at bottom-left
    
    Logic:
    1. S(0,0) -> (0,1) -> (0,2) [Choke Point].
       Path A: Cost 0. Length 2. 
       Max Battery = 2.
       Path A arrives at (0,2) with Battery 0.
       Path A cannot move to T(0,3). 
       BUT, Path A records min_damage[(0,2)] = 0.
       
    2. S(0,0) -> (1,0) -> (2,0)[Charger] -> ... -> (0,2).
       Path B goes down to charger, then snakes back to (0,2).
       Cost is High (walls).
       Battery is High (refilled).
       
    3. Collision:
       Path B arrives at (0,2). 
       Code checks: Is Path B Cost (High) < Path A Cost (0)? NO.
       Code PRUNES Path B.
       
    4. Failure:
       Path A is stuck (No battery).
       Path B is pruned (Logic error).
       Returns -1.
    """

    grid = [
        [0,   0,   0,   0],    # Row 0: S at (0,0), Target at (0,3)
        [100, 100, 100, 100],  # Row 1: Wall
        [-1,  100, 100, 100]   # Row 2: Charger at (2,0)
    ]
    
    # We need Path B to actually reach (0,2) from the bottom.
    # Let's open a corridor for Path B.
    # Grid Logic Refined:
    # S(0,0) -> (0,1) -> (0,2) is the Cheap Trap.
    # (2,0) is Charger.
    # We need a path from (2,0) to (0,2).
    # (2,0) -> (2,1) -> (2,2) -> (1,2) -> (0,2).
    # All these must be passable.
    
    grid = [
        [0,   0,   0,   0],    # (0,2) is the Choke. (0,3) is Target.
        [100, 100, 50,  100],  # (1,2) allows passage up
        [-1,  50,  50,  100]   # (2,0) -> (2,1) -> (2,2)
    ]
    
    # Max Battery = 2.
    # Path A: (0,0)->(0,1)->(0,2). Moves: 2. Batt Rem: 0.
    # At (0,2), needs to go to (0,3). Batt 0. FAILS.
    # Records min_damage[(0,2)] = 0.
    
    # Path B: (0,0)->(1,0)[100]->(2,0)[Charge].
    # (2,0)->(2,1)[50]->(2,2)[50]->(1,2)[50]->(0,2)[0].
    # Total Cost: 250+.
    # Moves from Charger: 4. 
    # Wait, Max Batt is 2. Path B will die on the way back?
    # We need Max Batt to be enough for B, but not for A.
    
    # Let's set Max Batt = 3.
    # Path A: (0,0)->(0,1)->(0,2). Moves 2. Batt Rem 1.
    # At (0,2), needs to go to (0,3). Batt 1.
    # Code: `if batt > 0`. 1 > 0. Success! 
    # Path A works if Batt=3. We need Path A to fail.
    
    # Constraint:
    # 1. Dist(Start -> Choke) = Max_Batt
    # 2. Dist(Charger -> Choke) <= Max_Batt
    
    # Let's push the Choke further away.
    # S(0,0).. (0,1).. (0,2).. (0,3)[Choke].. (0,4)[Target].
    # Dist 3 to Choke.
    # Max Batt 3.
    # Path A arrives Choke with Batt 0. Fails to reach Target.
    
    # Path B: S -> Down -> Charger -> ... -> Choke.
    # We need Charger to be close to Choke?
    # Charger at (1,3).
    # S(0,0) -> (1,0) -> (1,1) -> (1,2) -> (1,3)[C] -> (0,3)[Choke].
    # Dist S->C is 4. Max Batt 3. Path B dies reaching charger!
    
    # Okay, Charger must be reachable from Start.
    # S(0,0) -> (1,0)[C]. Dist 1.
    # From (1,0) to Choke (0,3). Dist 3.
    # Max Batt 3. Path B makes it!
    
    # FINAL GEOMETRY:
    # S(0,0) -> (0,1) -> (0,2) -> (0,3)[Choke] -> (0,4)[Target]
    # Path A Cost: 1+1+1+1 = 4.
    # Path B: (0,0) -> (1,0)[C] -> (1,1) -> (1,2) -> (1,3) -> (0,3).
    # Path B Cost: 100 + 100 + 100 + 100 + 0 = 400.
    
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
    # This is getting too complex to route B to (4,4).
    
    # SIMPLIFIED SQUARE 3x3
    # S(0,0), T(2,2).
    # Path A: (0,0)->(0,1)->(0,2). Choke is (0,2).
    # Then (0,2)->(1,2)->(2,2).
    # Length 4.
    # Max Batt 2.
    # Path A dies at (0,2). Records Cost ~2.
    
    # Path B: (0,0)->(1,0)[C].
    # (1,0)->(1,1)->(1,2)->(0,2).
    # Length from C: 3.
    # Max Batt 2. Path B dies too! (3 > 2).
    
    # We need Max Batt = 3.
    # Path A (Len 4). Arrives (1,2) with Batt 0. 
    #   (0,0)->(0,1)->(0,2)->(1,2).
    #   Dies at (1,2). Records Cost Low.
    # Path B (Len from C = 3?):
    #   (1,0)[C] -> (1,1) -> (1,2). Length 2.
    #   Arrives (1,2) with Batt 1.
    #   Can move to (2,2).
    #   Conflict at (1,2)!
    
    # 3x3 GRID FINAL:
    # [0,   0,   0]
    # [-1,  100, 0]  <-- (1,2) is the Choke Point
    # [100, 100, 0]  <-- (2,2) is Target
    
    grid_3x3 = [
        [0,   0,   0],    # A: (0,0)->(0,1)->(0,2)
        [-1,  100, 0],    # B: (0,0)->(1,0)[C]. Merge at (1,2).
        [100, 100, 0]     # Target (2,2)
    ]
    
    # Path A: (0,0)->(0,1)->(0,2)->(1,2).
    # Steps: 3.
    # Cost: 0.
    # Max Batt: 3.
    # At (1,2), Batt is 3 - 3 = 0.
    # Needs to go (1,2)->(2,2). Fails.
    # Records min_damage[(1,2)] = 0.
    
    # Path B: (0,0)->(1,0)[C]. Refill to 3.
    # Route: (1,0)->(1,1) is Wall? No, 100.
    # (1,0)->(1,1)[100]->(1,2)[0].
    # Steps from C: 2.
    # At (1,2), Batt is 3 - 2 = 1.
    # Cost: 100.
    # Can move to (2,2).
    
    # PRUNING:
    # Path B arrives (1,2). Cost 100.
    # Path A visited (1,2). Cost 0.
    # 100 >= 0. Prune.
    
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
