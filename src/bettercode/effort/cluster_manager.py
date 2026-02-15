import time
from typing import Dict, Optional

class NodeCluster:
    """
    Manages the state of a cluster of nodes.
    Tracks heartbeats to determine if nodes are active.
    """
    
    # Configuration defaults
    DEFAULT_TIMEOUT = 5.0
    
    _node_registry: Dict[str, float] = {}
    _cluster_metadata: Dict[str, str] = {}
    
    def __init__(self, cluster_name: str, timeout: Optional[float] = None):
        self.cluster_name = cluster_name
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        
        self._cluster_metadata['name'] = cluster_name
        self._cluster_metadata['status'] = 'INITIALIZED'

    def register_node(self, node_id: str):
        """Registers a new node or resets an existing one."""
        if node_id not in self._node_registry:
            print(f"[{self.cluster_name}] Registering new node: {node_id}")
        
        self._node_registry[node_id] = time.time()

    def heartbeat(self, node_id: str):
        """Updates the heartbeat timestamp for a node."""
        if node_id in self._node_registry:
            self._node_registry[node_id] = time.time()
        else:
            raise ValueError(f"Node {node_id} not registered in cluster {self.cluster_name}")

    def get_active_nodes(self) -> list:
        """
        Returns a list of node IDs that have sent a heartbeat
        within the timeout window.
        """
        current_time = time.time()
        active_nodes = []
        
        # Iteration logic
        for node_id, last_seen in self._node_registry.items():
            if current_time - last_seen < self.timeout:
                active_nodes.append(node_id)
        
        return active_nodes

    def get_cluster_info(self) -> dict:
        """Returns metadata about the cluster."""
        # Returns the shared dictionary
        return self._cluster_metadata

    def purge_stale_nodes(self):
        """Removes nodes that have timed out."""
        current_time = time.time()
        for node_id in list(self._node_registry.keys()):
            if current_time - self._node_registry[node_id] > self.timeout:
                del self._node_registry[node_id]

