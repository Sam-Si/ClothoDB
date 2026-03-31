"""
Consistent Hashing implementation for distributed routing.

Consistent hashing maps both nodes and keys to a hash ring. This ensures:
1. Minimal reorganization when nodes are added/removed
2. Even distribution of keys across nodes (with virtual nodes)
3. O(log n) lookup time for finding the responsible node

For replication, we return the top N nodes clockwise from the key's position.
"""

from __future__ import annotations

import bisect
import hashlib
from typing import List, Dict, Set, Optional, Callable, Tuple
from dataclasses import dataclass, field


def _default_hash(key: str) -> int:
    """Default hash function using MD5 (consistent across Python versions)."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16)


@dataclass(frozen=True)
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: str
    host: str
    port: int
    
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"
    
    def __hash__(self) -> int:
        return hash((self.node_id, self.host, self.port))
    
    def __repr__(self) -> str:
        return f"NodeInfo({self.node_id}@{self.address})"


class ConsistentHashRing:
    """
    Consistent Hash Ring for distributed routing.
    
    Each physical node is mapped to multiple virtual nodes (replicas) on the ring
    to ensure better load distribution.
    
    The ring is conceptually a circle where:
    - Each node is placed at multiple positions (virtual nodes)
    - Each key is placed at one position
    - The key is owned by the first node clockwise from its position
    
    For replication factor N, we return the first N unique nodes clockwise.
    
    Example:
    --------
    Hash Ring (0 to 2^160-1):
    
    Node A (vn0) ---- Node B (vn0) ---- Node C (vn0)
         |                  |                  |
    Node A (vn1)      Node B (vn1)      Node C (vn1)
    
    Key "user:123" hashes to position X.
    The first node clockwise from X owns the key.
    """
    
    def __init__(
        self,
        replicas: int = 150,  # Virtual nodes per physical node
        hash_func: Optional[Callable[[str], int]] = None
    ):
        """
        Initialize the hash ring.
        
        Args:
            replicas: Number of virtual nodes per physical node (higher = better distribution)
            hash_func: Custom hash function (defaults to MD5)
        """
        self.replicas = replicas
        self.hash_func = hash_func or _default_hash
        
        # Sorted list of hash positions (points on the ring)
        self._keys: List[int] = []
        
        # Map from hash position to NodeInfo
        self._ring: Dict[int, NodeInfo] = {}
        
        # Set of all physical nodes (for quick lookup)
        self._nodes: Set[NodeInfo] = set()
        
        # Map from node_id to NodeInfo
        self._node_map: Dict[str, NodeInfo] = {}
    
    def _hash(self, key: str) -> int:
        """Hash a key to a position on the ring."""
        return self.hash_func(key)
    
    def _virtual_node_key(self, node_id: str, replica: int) -> str:
        """Generate a unique key for a virtual node."""
        return f"{node_id}:{replica}"
    
    def add_node(self, node_info: NodeInfo) -> None:
        """
        Add a physical node to the ring.
        
        Creates `replicas` virtual nodes spread around the ring.
        """
        if node_info.node_id in self._node_map:
            raise ValueError(f"Node {node_info.node_id} already exists")
        
        self._nodes.add(node_info)
        self._node_map[node_info.node_id] = node_info
        
        # Create virtual nodes
        for i in range(self.replicas):
            virtual_key = self._virtual_node_key(node_info.node_id, i)
            position = self._hash(virtual_key)
            
            # Handle collision (extremely rare with good hash function)
            while position in self._ring:
                position = (position + 1) % (2 ** 128)
            
            self._ring[position] = node_info
            bisect.insort(self._keys, position)
    
    def remove_node(self, node_id: str) -> None:
        """
        Remove a physical node from the ring.
        
        All its virtual nodes are removed, and keys are redistributed
        to the next nodes clockwise.
        """
        if node_id not in self._node_map:
            raise ValueError(f"Node {node_id} not found")
        
        node_info = self._node_map[node_id]
        self._nodes.discard(node_info)
        del self._node_map[node_id]
        
        # Remove all virtual nodes
        for i in range(self.replicas):
            virtual_key = self._virtual_node_key(node_id, i)
            position = self._hash(virtual_key)
            
            if position in self._ring:
                del self._ring[position]
                self._keys.remove(position)
    
    def get_node(self, key: str) -> Optional[NodeInfo]:
        """
        Get the primary node responsible for a key.
        
        Returns the first node clockwise from the key's position.
        """
        if not self._keys:
            return None
        
        position = self._hash(key)
        
        # Binary search for the first node >= position
        idx = bisect.bisect_right(self._keys, position)
        
        if idx == len(self._keys):
            # Wrap around to the first node
            idx = 0
        
        node_pos = self._keys[idx]
        return self._ring[node_pos]
    
    def get_nodes(self, key: str, n: int = 3) -> List[NodeInfo]:
        """
        Get the top N nodes responsible for a key (for replication).
        
        Returns up to N unique nodes clockwise from the key's position.
        If there are fewer than N nodes in the ring, returns all nodes.
        
        Args:
            key: The key to look up
            n: Number of replicas (default 3 for N=3 replication)
        
        Returns:
            List of up to N unique NodeInfo objects
        """
        if not self._keys:
            return []
        
        if n <= 0:
            return []
        
        position = self._hash(key)
        
        # Binary search for the starting position
        idx = bisect.bisect_right(self._keys, position)
        
        # Collect N unique nodes
        result: List[NodeInfo] = []
        seen: Set[str] = set()
        
        for _ in range(len(self._keys)):
            if len(result) >= n:
                break
            
            if idx >= len(self._keys):
                idx = 0
            
            node_pos = self._keys[idx]
            node = self._ring[node_pos]
            
            if node.node_id not in seen:
                result.append(node)
                seen.add(node.node_id)
            
            idx += 1
        
        return result
    
    def get_all_nodes(self) -> List[NodeInfo]:
        """Get all physical nodes in the ring."""
        return list(self._nodes)
    
    def node_count(self) -> int:
        """Get the number of physical nodes."""
        return len(self._nodes)
    
    def virtual_node_count(self) -> int:
        """Get the total number of virtual nodes."""
        return len(self._keys)
    
    def get_node_keys(self, node_id: str) -> List[str]:
        """
        Get all keys that would be assigned to a specific node.
        
        This is useful for key migration when nodes join/leave.
        """
        # This is an expensive operation - in production, you'd maintain
        # an index. Here we just return an empty list as placeholder.
        return []
    
    def get_ring_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of virtual nodes across physical nodes.
        
        Returns a map of node_id -> virtual node count.
        """
        distribution: Dict[str, int] = {}
        for node in self._ring.values():
            distribution[node.node_id] = distribution.get(node.node_id, 0) + 1
        return distribution
    
    def __contains__(self, node_id: str) -> bool:
        """Check if a node is in the ring."""
        return node_id in self._node_map
    
    def __len__(self) -> int:
        """Return the number of physical nodes."""
        return len(self._nodes)
    
    def __repr__(self) -> str:
        return f"ConsistentHashRing(nodes={self.node_count()}, vnodes={self.virtual_node_count()})"


class Router:
    """
    High-level router for distributed key operations.
    
    Provides APIs for:
    - Getting nodes responsible for a key
    - Routing operations to appropriate nodes
    - Managing the cluster topology
    """
    
    DEFAULT_REPLICATION_FACTOR = 3
    
    def __init__(
        self,
        replication_factor: int = DEFAULT_REPLICATION_FACTOR,
        virtual_nodes: int = 150
    ):
        """
        Initialize the router.
        
        Args:
            replication_factor: Number of replicas for each key (N)
            virtual_nodes: Number of virtual nodes per physical node
        """
        self.replication_factor = replication_factor
        self.ring = ConsistentHashRing(replicas=virtual_nodes)
    
    def add_server(self, node_id: str, host: str, port: int) -> NodeInfo:
        """
        Add a server to the routing table.
        
        Returns the NodeInfo for the added server.
        """
        node_info = NodeInfo(node_id=node_id, host=host, port=port)
        self.ring.add_node(node_info)
        return node_info
    
    def remove_server(self, node_id: str) -> None:
        """Remove a server from the routing table."""
        self.ring.remove_node(node_id)
    
    def get_primary_node(self, key: str) -> Optional[NodeInfo]:
        """Get the primary (coordinator) node for a key."""
        return self.ring.get_node(key)
    
    def get_replica_nodes(self, key: str) -> List[NodeInfo]:
        """
        Get all nodes that should store a replica of the key.
        
        Returns up to replication_factor nodes.
        """
        return self.ring.get_nodes(key, n=self.replication_factor)
    
    def route_key(self, key: str) -> Dict[str, any]:
        """
        Get complete routing information for a key.
        
        Returns a dict with:
        - key: the key being routed
        - primary: the primary node
        - replicas: list of replica nodes (including primary as first)
        - replication_factor: the configured replication factor
        """
        replicas = self.get_replica_nodes(key)
        
        return {
            "key": key,
            "primary": replicas[0] if replicas else None,
            "replicas": replicas,
            "replication_factor": min(self.replication_factor, len(replicas)),
            "success": len(replicas) > 0
        }
    
    def get_all_servers(self) -> List[NodeInfo]:
        """Get all servers in the cluster."""
        return self.ring.get_all_nodes()
    
    def get_server_count(self) -> int:
        """Get the number of servers in the cluster."""
        return self.ring.node_count()
    
    def get_distribution_stats(self) -> Dict:
        """
        Get statistics about the key distribution.
        
        Returns info about virtual node distribution.
        """
        distribution = self.ring.get_ring_distribution()
        
        if not distribution:
            return {"total_nodes": 0, "virtual_nodes": 0}
        
        vnodes = list(distribution.values())
        
        return {
            "total_nodes": len(distribution),
            "virtual_nodes": sum(vnodes),
            "avg_vnodes_per_node": sum(vnodes) / len(vnodes),
            "min_vnodes": min(vnodes),
            "max_vnodes": max(vnodes),
            "distribution": distribution
        }


def create_5_server_cluster(
    base_port: int = 8000,
    replication_factor: int = 3
) -> Router:
    """
    Create a router with 5 servers on consecutive ports.
    
    This is a convenience function for testing/demo purposes.
    
    Args:
        base_port: Starting port number (default 8000)
        replication_factor: Replication factor (default 3)
    
    Returns:
        Configured Router with 5 servers
    """
    router = Router(replication_factor=replication_factor)
    
    for i in range(5):
        node_id = f"server-{i+1}"
        port = base_port + i
        router.add_server(node_id, "127.0.0.1", port)
    
    return router
