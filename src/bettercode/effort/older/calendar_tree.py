class Interval:
    def __init__(self, start, end, id_val):
        self.start = start
        self.end = end
        self.id = id_val

class TimeNode:
    def __init__(self, range_start, range_end):
        self.range_start = range_start
        self.range_end = range_end
        self.left = None
        self.right = None
        # Stores intervals that exactly match this node's full range
        # or cannot be pushed further down (though logic below pushes aggressively)
        self.intervals = []

    def insert(self, interval):
        # If the interval covers the entire range of this node, store it here.
        if interval.start <= self.range_start and interval.end >= self.range_end:
            self.intervals.append(interval)
            return

        mid = (self.range_start + self.range_end) // 2

        # PARTITION LOGIC (The Bug is here)
        # It looks like standard Binary Space Partitioning
        if interval.end <= mid:
            # Fits entirely in left half
            if not self.left:
                self.left = TimeNode(self.range_start, mid)
            self.left.insert(interval)
        else:
            # Fits in right half OR straddles the middle
            if not self.right:
                self.right = TimeNode(mid, self.range_end)
            self.right.insert(interval)

    def find_overlap(self, query_interval):
        # 1. Check against intervals stored strictly at this node
        for stored in self.intervals:
            if max(self.range_start, query_interval.start) < min(self.range_end, query_interval.end):
                # Standard overlap check: start1 < end2 and start2 < end1
                if stored.start < query_interval.end and query_interval.start < stored.end:
                    return stored.id

        mid = (self.range_start + self.range_end) // 2

        # 2. Recurse
        found = None
        
        # If query touches left half, check left child
        if self.left and query_interval.start < mid:
            found = self.left.find_overlap(query_interval)
            if found is not None: return found

        # If query touches right half, check right child
        if self.right and query_interval.end > mid:
            found = self.right.find_overlap(query_interval)
            
        return found

class MaintenanceScheduler:
    def __init__(self, max_time=100):
        self.root = TimeNode(0, max_time)

    def schedule(self, start, end, job_id):
        """
        Returns the ID of a conflicting job if one exists.
        Otherwise, adds the job and returns None.
        """
        new_job = Interval(start, end, job_id)
        
        # Check collision
        conflict = self.root.find_overlap(new_job)
        if conflict is not None:
            return conflict
        
        # No collision, add it
        self.root.insert(new_job)
        return None
