from warehouse import InventoryManager
def test_single_warehouse_operations():
    wh = InventoryManager("WH-01")
    assert wh.add_stock("SKU-1", 10) is True
    assert wh.get_stock_level("SKU-1") == 10
    assert wh.remove_stock("SKU-1", 5) is True
    assert wh.get_stock_level("SKU-1") == 5
def test_capacity_limits():
    wh = InventoryManager("WH-02", capacity=20)
    assert wh.add_stock("SKU-HEAVY", 15) is True
    assert wh.add_stock("SKU-HEAVY", 10) is False # Over capacity
    assert wh.get_stock_level("SKU-HEAVY") == 15
def test_isolation_between_warehouses():
    # This is the critical test that fails due to the class attribute bug
    wh_east = InventoryManager("WH-EAST")
    wh_west = InventoryManager("WH-WEST")
    
    # Add items to East
    wh_east.add_stock("APPLE", 100)
    
    # Check West is empty
    # The bug causes West to share the _inventory dict with East
    # So West will see the apples!
    assert wh_west.get_stock_level("APPLE") == 0 
    
    # Add different items to West
    wh_west.add_stock("BANANA", 50)
    
    # East should not have bananas
    assert wh_east.get_stock_level("BANANA") == 0
    
def test_independent_capacity():
    # Even capacity checks might be affected if total inventory is shared
    wh_limited = InventoryManager("WH-LIM", capacity=10)
    wh_other = InventoryManager("WH-OTHER")
    
    # Fill the limited one to capacity
    wh_limited.add_stock("ITEM", 10)
    
    # The other warehouse should be empty and have space (if it had capacity set)
    # But if they share state, the sum() in add_stock might fail or behave weirdly
    # Let's strictly test isolation
    assert wh_other.get_stock_level("ITEM") == 0
    assert wh_other.add_stock("NEW", 5) is True
