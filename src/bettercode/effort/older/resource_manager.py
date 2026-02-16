class ResourceManager:
    def __init__(self):
        self._resources = {} # id -> {'data': ..., 'ref_count': ...}
        self.on_destroy_callback = None

    def acquire(self, resource_id):
        if resource_id not in self._resources:
            print(f"Loading {resource_id}...")
            self._resources[resource_id] = {'data': f"DATA_{resource_id}", 'ref_count': 0}
        self._resources[resource_id]['ref_count'] += 1
        return self._resources[resource_id]['data']

    def release(self, resource_id):
        if resource_id not in self._resources: return

        res = self._resources[resource_id]
        res['ref_count'] -= 1
        
        if res['ref_count'] == 0:
            # Notify listeners that cleanup is about to happen
            if self.on_destroy_callback:
                self.on_destroy_callback(resource_id)
            
            # Unload the resource to free memory
            print(f"Deleting {resource_id}...")
            del self._resources[resource_id]

    def is_loaded(self, resource_id):
        return resource_id in self._resources
