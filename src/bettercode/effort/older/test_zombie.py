import pytest
from resource_manager import ResourceManager

def test_basic_lifecycle():
    rm = ResourceManager()
    rm.acquire("A") # Ref 1
    rm.acquire("A") # Ref 2
    rm.release("A") # Ref 1
    assert rm.is_loaded("A")
    rm.release("A") # Ref 0 -> Unload
    assert not rm.is_loaded("A")

def test_resurrection_bug():
    """
    THE KILLER TEST:
    A subsystem 'resurrects' the resource during the destroy callback.
    The manager fails to notice the resurrection and deletes it anyway.
    """
    rm = ResourceManager()
    
    # 1. Define a callback that tries to keep the object alive
    def attempt_resurrection(res_id):
        print(f"Callback: Oh no, {res_id} is dying! Quick, acquire it again!")
        rm.acquire(res_id)
        # At this point, ref_count should be back to 1.
        # The resource should remain in memory.

    rm.on_destroy_callback = attempt_resurrection
    
    # 2. Lifecycle
    rm.acquire("ZombieTexture") # Ref = 1
    
    # 3. Release triggers the bug
    # Count -> 0 -> Callback -> Acquire (Count -> 1) -> Buggy Code Deletes it.
    rm.release("ZombieTexture")
    
    # 4. Verification
    # If the bug is present, is_loaded returns False (it was deleted).
    # If the bug is fixed, it should be True (it was resurrected).
    assert rm.is_loaded("ZombieTexture"), \
        "Logic Error: Resource was deleted even though it was re-acquired during the destroy callback!"
    
    # 5. Verify integrity (Double check)
    # If we acquire it again, it should NOT print "Loading..." if it was correctly preserved.
    # (Checking internal state for demonstration)
    if "ZombieTexture" in rm._resources:
        assert rm._resources["ZombieTexture"]['ref_count'] == 1
