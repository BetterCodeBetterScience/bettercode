import time
from typing import Dict, List, Optional
class AuditLogMixin:
    """Provides audit logging capabilities."""
    def log_event(self, event_type: str, data: dict):
        # simulate logging to an external system
        timestamp = time.time()
        # In a real system, this would go to a file or socket
        # print(f"[{timestamp}] {event_type}: {data}")
        pass
class InventoryManager(AuditLogMixin):
    """
    Manages inventory for a specific warehouse location.
    Uses a local storage dict to track item counts.
    """
    
    # Bug: _inventory is a class attribute (mutable), shared across all instances.
    # It should be an instance attribute initialized in __init__.
    _inventory: Dict[str, int] = {}
    
    def __init__(self, location_id: str, capacity: int = 100):
        self.location_id = location_id
        self.capacity = capacity
        # The developer forgot to initialize self._inventory = {} here.
        # They likely thought the type hint above handled it, or just missed it.
    def add_stock(self, item_sku: str, quantity: int) -> bool:
        """Adds stock for a specific item. Returns False if over capacity."""
        if not isinstance(quantity, int) or quantity < 0:
            raise ValueError("Quantity must be a non-negative integer")
            
        current_total = sum(self._inventory.values())
        if current_total + quantity > self.capacity:
            self.log_event("CAPACITY_EXCEEDED", {"location": self.location_id, "item": item_sku})
            return False
            
        self._inventory[item_sku] = self._inventory.get(item_sku, 0) + quantity
        self.log_event("STOCK_ADDED", {"location": self.location_id, "item": item_sku, "qty": quantity})
        return True
    def remove_stock(self, item_sku: str, quantity: int) -> bool:
        """Removes stock. Returns False if insufficient stock."""
        if item_sku not in self._inventory or self._inventory[item_sku] < quantity:
            self.log_event("INSUFFICIENT_STOCK", {"location": self.location_id, "item": item_sku})
            return False
        
        self._inventory[item_sku] -= quantity
        if self._inventory[item_sku] == 0:
            del self._inventory[item_sku]
            
        self.log_event("STOCK_REMOVED", {"location": self.location_id, "item": item_sku, "qty": quantity})
        return True
    def get_stock_level(self, item_sku: str) -> int:
        return self._inventory.get(item_sku, 0)
        
    def get_all_inventory(self) -> Dict[str, int]:
        # Returns a copy to prevent direct mutation
        return self._inventory.copy()
    def __repr__(self):
        return f"<InventoryManager location={self.location_id} items={len(self._inventory)}>"
