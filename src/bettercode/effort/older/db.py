import threading

class VersionedValue:
    def __init__(self, value, version):
        self.value = value
        self.version = version

class KVStore:
    def __init__(self):
        self.store = {} 
        self.global_clock = 0
        self.lock = threading.Lock()

    def begin_transaction(self):
        return Transaction(self)

    def commit_transaction(self, tx):
        with self.lock:
            # VALIDATION: Check for Write-Write Conflicts
            # Ensure keys we are writing haven't changed since tx started
            for key in tx.write_set:
                if key in self.store:
                    if self.store[key].version > tx.start_clock:
                        return False # Abort
            
            # Apply Writes
            new_clock = self.global_clock + 1
            for key, val in tx.write_set.items():
                self.store[key] = VersionedValue(val, new_clock)
            self.global_clock = new_clock
            return True

class Transaction:
    def __init__(self, db):
        self.db = db
        with db.lock:
            self.start_clock = db.global_clock
        self.read_set = {}
        self.write_set = {}

    def read(self, key):
        if key in self.write_set: return self.write_set[key]
        if key in self.read_set: return self.read_set[key]
        
        with self.db.lock:
            if key in self.db.store:
                self.read_set[key] = self.db.store[key].value
                return self.read_set[key]
        return None

    def write(self, key, value):
        self.write_set[key] = value

    def commit(self):
        return self.db.commit_transaction(self)
