import heapq

def solve_rover_path(grid, max_battery):
    n = len(grid)
    start = (0, 0)
    target = (n - 1, n - 1)
    
    pq = [(0, max_battery, 0, 0)]
    
    min_damage = {} 
    
    while pq:
        dmg, batt, r, c = heapq.heappop(pq)
        
        if (r, c) == target:
            return dmg
        
        if (r, c) in min_damage and min_damage[(r, c)] <= dmg:
            continue
        
        min_damage[(r, c)] = dmg
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < n and 0 <= nc < n:
                if batt > 0:
                    new_batt = batt - 1
                    cell_val = grid[nr][nc]
                    new_dmg = dmg
                    
                    if cell_val == -1: # Charger
                        new_batt = max_battery
                    else:
                        new_dmg += cell_val
                        
                    heapq.heappush(pq, (new_dmg, new_batt, nr, nc))
                    
    return -1
