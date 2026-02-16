import pytest
import threading
from db import KVStore

def test_basic_commit():
    db = KVStore()
    t1 = db.begin_transaction()
    t1.write("x", 100)
    assert t1.commit() is True
    
    t2 = db.begin_transaction()
    assert t2.read("x") == 100

def test_lost_update_prevention():
    """
    Ensures standard Write-Write conflicts are caught.
    Tx1 reads X. Tx2 reads X. Tx1 writes X. Tx2 writes X.
    """
    db = KVStore()
    
    # Initialize
    setup = db.begin_transaction()
    setup.write("x", 10)
    setup.commit()
    
    # Concurrent Tx
    t1 = db.begin_transaction()
    t2 = db.begin_transaction()
    
    v1 = t1.read("x")
    v2 = t2.read("x")
    
    t1.write("x", v1 + 1)
    t2.write("x", v2 + 1)
    
    assert t1.commit() is True
    # t2 should fail because 'x' changed since t2 started
    assert t2.commit() is False 

def test_write_skew_anomaly():
    """
    THE KILLER TEST.
    
    Scenario:
    Two accounts, A and B.
    Invariant: A + B >= 0.
    Initial: A=100, B=100.
    
    Tx1: Withdraw 150 from A. (Checks 100+100=200 >= 150. Safe).
    Tx2: Withdraw 150 from B. (Checks 100+100=200 >= 150. Safe).
    
    Since Tx1 only writes A, and Tx2 only writes B,
    a simple "Write Set Validation" (Snapshot Isolation) 
    will allow BOTH to commit.
    
    Result: A=-50, B=-50. Sum = -100. Invariant Broken.
    """
    db = KVStore()
    
    # Setup
    init = db.begin_transaction()
    init.write("A", 100)
    init.write("B", 100)
    init.commit()
    
    # Run concurrent transactions
    t1 = db.begin_transaction()
