import bisect

class PerimeterMonitor:
    def __init__(self):
        self.cameras = []

    def add_camera(self, angle):
        bisect.insort(self.cameras, angle)

    def is_responsible(self, camera_angle, event_angle):
        if camera_angle not in self.cameras: return False

        idx = self.cameras.index(camera_angle)
        
        # Determine the start of this camera's sector (the previous camera)
        if idx == 0:
            sector_start = self.cameras[-1]
        else:
            sector_start = self.cameras[idx - 1]
            
        sector_end = camera_angle

        # Check if event is within this sector
        if sector_start < event_angle <= sector_end:
            return True
        
        return False
