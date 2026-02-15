import pytest
import time
from cluster_manager import NodeCluster

def test_single_cluster_behavior():
    """
    This test passes fine. 
    It obscures the bug because basic functionality seems to work.
    """
    cluster_a = NodeCluster("Cluster-A", timeout=10.0)
    
    cluster_a.register_node("node-1")
    cluster_a.register_node("node-2")
    
    # Small sleep to simulate time passing, but within timeout
    time.sleep(0.1)
    
    active = cluster_a.get_active_nodes()
    assert len(active) == 2
    assert "node-1" in active
    
    info = cluster_a.get_cluster_info()
    assert info['name'] == "Cluster-A"
    assert info['status'] == "INITIALIZED"

def test_multi_cluster_isolation():
    """
    This test verifies that different cluster instances maintain separate states.
    This is where the bug manifests.
    """
    # Create Cluster B
    cluster_b = NodeCluster("Cluster-B", timeout=5.0)
    
    # Create Cluster C
    cluster_c = NodeCluster("Cluster-C", timeout=5.0)
    
    # Register nodes in Cluster B
    cluster_b.register_node("node-b1")
    cluster_b.register_node("node-b2")
    
    # Register nodes in Cluster C
    cluster_c.register_node("node-c1")
    
    # --- ASSERTION FAILURE POINT 1: Node Registry Isolation ---
    # Cluster C should only have 'node-c1'.
    # However, because _node_registry is a class attribute, it sees B's nodes too.
    c_nodes = cluster_c.get_active_nodes()
    print(f"Cluster C sees nodes: {c_nodes}")
    assert len(c_nodes) == 1, f"Expected 1 node in Cluster C, found {len(c_nodes)}"
    
    # --- ASSERTION FAILURE POINT 2: Metadata Isolation ---
    # Cluster B's metadata should reflect Cluster B.
    # Because _cluster_metadata is shared, Cluster C's init overwrote the 'name'.
    b_info = cluster_b.get_cluster_info()
    assert b_info['name'] == "Cluster-B", f"Expected name 'Cluster-B', got {b_info['name']}"

def test_node_modification_isolation():
    """
    Tests that modifying nodes in one cluster doesn't affect another.
    """
    cluster_x = NodeCluster("Cluster-X")
    cluster_y = NodeCluster("Cluster-Y")
    
    cluster_x.register_node("shared-node-id")
    
    # Even though the ID is the same, it should be isolated per cluster.
    # But since they share the registry dict, the key is just overwritten.
    # In this specific case, length is 1 in the shared dict.
    
    # However, the previous test already failed. This test adds checks for logic flow.
    # Let's check if Y sees X's node.
    y_nodes = cluster_y.get_active_nodes()
    assert "shared-node-id" not in y_nodes, "Cluster Y should not see Cluster X's nodes"

def test_attribute_memory():
    """
    Tests if a new instance retains state from old instances improperly.
    """
    cluster_old = NodeCluster("Old-Cluster")
    cluster_old.register_node("ghost-node")
    
    # Destroy reference to old cluster
    del cluster_old
    
    # Create a totally new cluster
    cluster_new = NodeCluster("New-Cluster")
    
    # The new cluster should be empty.
    # But because _node_registry is a class attribute, 'ghost-node' persists.
    new_nodes = cluster_new.get_active_nodes()
    assert len(new_nodes) == 0, f"New cluster should be empty, but found {new_nodes}"

