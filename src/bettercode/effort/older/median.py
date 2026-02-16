import heapq
from collections import defaultdict

class RollingMedian:
    def __init__(self):
        # Lower half (Max Heap). Stored as negative numbers for heapq.
        self.lo = [] 
        # Upper half (Min Heap).
        self.hi = []
        # Lazy deletion tracker
        self.deleted = defaultdict(int)

    def add_latency(self, val):
        # Standard Two-Heap Insert
        if not self.lo or val <= -self.lo[0]:
            heapq.heappush(self.lo, -val)
        else:
            heapq.heappush(self.hi, val)
        
        self._rebalance()

    def remove_latency(self, val):
        # Lazy Deletion: Just mark it. Don't scan heaps (that would be O(N)).
        self.deleted[val] += 1

    def get_median(self):
        # Prune ghosts from the top before peeking
        self._clean_top()

        if not self.lo and not self.hi:
            return 0.0

        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        elif len(self.hi) > len(self.lo):
            return self.hi[0]
        else:
            return (-self.lo[0] + self.hi[0]) / 2.0

    def _rebalance(self):
        # Keep sizes equal or lo having +1
        
        if len(self.lo) > len(self.hi) + 1:
            val = -heapq.heappop(self.lo)
            heapq.heappush(self.hi, val)
        elif len(self.hi) > len(self.lo):
            val = heapq.heappop(self.hi)
            heapq.heappush(self.lo, -val)
            
        # Note: We do not clean inside rebalance to avoid recursion loops
        # or O(N) behavior, assuming get_median handles cleanup.

    def _clean_top(self):
        # Standard Lazy Removal: Clean tops of heaps
        while self.lo and self.deleted[-self.lo[0]] > 0:
            val = -heapq.heappop(self.lo)
            self.deleted[val] -= 1
            
        while self.hi and self.deleted[self.hi[0]] > 0:
            val = heapq.heappop(self.hi)
            self.deleted[val] -= 1
