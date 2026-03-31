"""
Tests for consistent hashing implementation.

These tests verify:
1. Basic ring operations (add/remove nodes)
2. Key routing consistency
3. Virtual node distribution
4. Replication (top N nodes)
5. Minimal reorganization on node changes
"""

import pytest
from hypothesis import given, strategies as st, settings

from clotho.consistent_hash import (
    ConsistentHashRing,
    Router,
    NodeInfo,
    create_5_server_cluster
)
from clotho.server import (
    get_top3_nodes_for_key,
    get_top3_nodes_api
)


class TestNodeInfo:
    """Tests for the NodeInfo dataclass."""
    
    def test_node_info_creation(self):
        """Should create NodeInfo with correct attributes."""
        node = NodeInfo(node_id="server-1", host="127.0.0.1", port=8000)
        assert node.node_id == "server-1"
        assert node.host == "127.0.0.1"
        assert node.port == 8000
        assert node.address == "127.0.0.1:8000"
    
    def test_node_info_immutable(self):
        """NodeInfo should be immutable (frozen dataclass)."""
        node = NodeInfo(node_id="server-1", host="127.0.0.1", port=8000)
        with pytest.raises(Exception):
            node.port = 8001
    
    def test_node_info_hashable(self):
        """NodeInfo should be hashable for use in sets/dicts."""
        node1 = NodeInfo(node_id="server-1", host="127.0.0.1", port=8000)
        node2 = NodeInfo(node_id="server-1", host="127.0.0.1", port=8000)
        node3 = NodeInfo(node_id="server-2", host="127.0.0.1", port=8001)
        
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)


class TestConsistentHashRing:
    """Tests for the ConsistentHashRing class."""
    
    def test_empty_ring_returns_none(self):
        """Empty ring should return None for any key."""
        ring = ConsistentHashRing()
        assert ring.get_node("any-key") is None
        assert ring.get_nodes("any-key", n=3) == []
    
    def test_add_node_increases_vnode_count(self):
        """Adding a node should create virtual nodes."""
        ring = ConsistentHashRing(replicas=100)
        node = NodeInfo(node_id="server-1", host="127.0.0.1", port=8000)
        
        ring.add_node(node)
        
        assert ring.node_count() == 1
        assert ring.virtual_node_count() == 100
    
    def test_add_duplicate_node_raises_error(self):
        """Adding a node with duplicate ID should raise error."""
        ring = ConsistentHashRing()
        node1 = NodeInfo(node_id="server-1", host="127.0.0.1", port=8000)
        node2 = NodeInfo(node_id="server-1", host="127.0.0.1", port=8001)
        
        ring.add_node(node1)
        with pytest.raises(ValueError):
            ring.add_node(node2)
    
    def test_remove_node_decreases_count(self):
        """Removing a node should decrease counts."""
        ring = ConsistentHashRing(replicas=50)
        node = NodeInfo(node_id="server-1", host="127.0.0.1", port=8000)
        
        ring.add_node(node)
        assert ring.node_count() == 1
        
        ring.remove_node("server-1")
        assert ring.node_count() == 0
        assert ring.virtual_node_count() == 0
    
    def test_remove_nonexistent_node_raises_error(self):
        """Removing a non-existent node should raise error."""
        ring = ConsistentHashRing()
        with pytest.raises(ValueError):
            ring.remove_node("nonexistent")
    
    def test_get_node_returns_consistent_result(self):
        """Same key should always map to same node."""
        ring = ConsistentHashRing()
        for i in range(5):
            ring.add_node(NodeInfo(f"server-{i}", "127.0.0.1", 8000 + i))
        
        # Same key should always return same node
        node1 = ring.get_node("test-key")
        node2 = ring.get_node("test-key")
        node3 = ring.get_node("test-key")
        
        assert node1 == node2 == node3
    
    def test_get_nodes_returns_multiple_unique(self):
        """Should return N unique nodes for replication."""
        ring = ConsistentHashRing()
        for i in range(5):
            ring.add_node(NodeInfo(f"server-{i}", "127.0.0.1", 8000 + i))
        
        nodes = ring.get_nodes("test-key", n=3)
        
        assert len(nodes) == 3
        # All should be unique
        assert len(set(n.node_id for n in nodes)) == 3
    
    def test_get_nodes_returns_all_if_fewer_than_n(self):
        """Should return all nodes if fewer than N exist."""
        ring = ConsistentHashRing()
        ring.add_node(NodeInfo("server-1", "127.0.0.1", 8000))
        ring.add_node(NodeInfo("server-2", "127.0.0.1", 8001))
        
        nodes = ring.get_nodes("test-key", n=5)
        
        assert len(nodes) == 2
    
    def test_different_keys_distribute_across_nodes(self):
        """Different keys should distribute across nodes."""
        ring = ConsistentHashRing()
        for i in range(5):
            ring.add_node(NodeInfo(f"server-{i}", "127.0.0.1", 8000 + i))
        
        # Map many keys and count distribution
        node_counts = {}
        for i in range(1000):
            node = ring.get_node(f"key-{i}")
            node_counts[node.node_id] = node_counts.get(node.node_id, 0) + 1
        
        # All nodes should have some keys (with high probability)
        assert len(node_counts) == 5
        
        # Distribution should be reasonably even (within factor of 3)
        counts = list(node_counts.values())
        assert max(counts) / min(counts) < 3
    
    def test_node_removal_minimal_reorganization(self):
        """Removing a node should only affect its keys."""
        ring = ConsistentHashRing()
        for i in range(5):
            ring.add_node(NodeInfo(f"server-{i}", "127.0.0.1", 8000 + i))
        
        # Map keys before removal
        original_mapping = {}
        for i in range(100):
            key = f"key-{i}"
            original_mapping[key] = ring.get_node(key).node_id
        
        # Remove one node
        ring.remove_node("server-2")
        
        # Check how many keys changed
        changed = 0
        for key, original_node in original_mapping.items():
            new_node = ring.get_node(key).node_id
            if original_node != new_node:
                changed += 1
                # Should only move to a different node, not to server-2
                assert new_node != "server-2"
        
        # Only ~20% of keys should have changed (1/5 of the ring)
        assert changed < 30  # Allow some variance
    
    def test_contains_operator(self):
        """__contains__ should check node existence."""
        ring = ConsistentHashRing()
        node = NodeInfo("server-1", "127.0.0.1", 8000)
        ring.add_node(node)
        
        assert "server-1" in ring
        assert "server-2" not in ring
    
    def test_len_operator(self):
        """__len__ should return physical node count."""
        ring = ConsistentHashRing()
        assert len(ring) == 0
        
        ring.add_node(NodeInfo("server-1", "127.0.0.1", 8000))
        assert len(ring) == 1


class TestRouter:
    """Tests for the Router class."""
    
    def test_router_creation(self):
        """Should create router with default settings."""
        router = Router()
        assert router.replication_factor == 3
        assert router.get_server_count() == 0
    
    def test_add_server(self):
        """Should add server and return NodeInfo."""
        router = Router()
        node = router.add_server("server-1", "127.0.0.1", 8000)
        
        assert node.node_id == "server-1"
        assert node.host == "127.0.0.1"
        assert node.port == 8000
        assert router.get_server_count() == 1
    
    def test_get_primary_node(self):
        """Should return primary node for key."""
        router = Router()
        for i in range(5):
            router.add_server(f"server-{i}", "127.0.0.1", 8000 + i)
        
        primary = router.get_primary_node("test-key")
        assert primary is not None
        assert primary.node_id.startswith("server-")
    
    def test_get_replica_nodes(self):
        """Should return replica nodes (including primary)."""
        router = Router(replication_factor=3)
        for i in range(5):
            router.add_server(f"server-{i}", "127.0.0.1", 8000 + i)
        
        replicas = router.get_replica_nodes("test-key")
        
        assert len(replicas) == 3
        # Primary should be first
        assert replicas[0] == router.get_primary_node("test-key")
    
    def test_route_key_returns_complete_info(self):
        """route_key should return complete routing information."""
        router = Router(replication_factor=3)
        for i in range(5):
            router.add_server(f"server-{i}", "127.0.0.1", 8000 + i)
        
        info = router.route_key("my-key")
        
        assert info["key"] == "my-key"
        assert info["success"] is True
        assert info["replication_factor"] == 3
        assert info["primary"] is not None
        assert len(info["replicas"]) == 3
    
    def test_get_distribution_stats(self):
        """Should return distribution statistics."""
        router = Router(virtual_nodes=100)
        for i in range(5):
            router.add_server(f"server-{i}", "127.0.0.1", 8000 + i)
        
        stats = router.get_distribution_stats()
        
        assert stats["total_nodes"] == 5
        assert stats["virtual_nodes"] == 500
        assert stats["avg_vnodes_per_node"] == 100.0


class TestCreate5ServerCluster:
    """Tests for the create_5_server_cluster convenience function."""
    
    def test_creates_5_servers(self):
        """Should create exactly 5 servers."""
        router = create_5_server_cluster(base_port=8000)
        
        assert router.get_server_count() == 5
        
        servers = router.get_all_servers()
        assert len(servers) == 5
    
    def test_servers_on_consecutive_ports(self):
        """Servers should be on consecutive ports starting from base."""
        router = create_5_server_cluster(base_port=9000)
        
        servers = router.get_all_servers()
        ports = sorted(s.port for s in servers)
        
        assert ports == [9000, 9001, 9002, 9003, 9004]
    
    def test_server_naming(self):
        """Servers should be named server-1 through server-5."""
        router = create_5_server_cluster()
        
        servers = router.get_all_servers()
        names = sorted(s.node_id for s in servers)
        
        assert names == ["server-1", "server-2", "server-3", "server-4", "server-5"]


class TestGetTop3NodesAPI:
    """Tests for the get_top3_nodes_api function."""
    
    def test_returns_top_3_nodes(self):
        """Should return exactly 3 nodes for any key."""
        result = get_top3_nodes_api("test-key")
        
        assert result["key"] == "test-key"
        assert len(result["top_3_nodes"]) == 3
        assert result["total_nodes_in_cluster"] == 5
        assert result["replication_factor"] == 3
    
    def test_first_node_is_primary(self):
        """First node should be marked as primary."""
        result = get_top3_nodes_api("test-key")
        
        assert result["top_3_nodes"][0]["is_primary"] is True
        assert result["top_3_nodes"][1]["is_primary"] is False
        assert result["top_3_nodes"][2]["is_primary"] is False
    
    def test_nodes_have_required_fields(self):
        """Each node should have all required fields."""
        result = get_top3_nodes_api("test-key")
        
        for node in result["top_3_nodes"]:
            assert "rank" in node
            assert "node_id" in node
            assert "host" in node
            assert "port" in node
            assert "address" in node
            assert "is_primary" in node
    
    def test_different_keys_may_have_different_top3(self):
        """Different keys should potentially have different top 3."""
        result1 = get_top3_nodes_api("key-a")
        result2 = get_top3_nodes_api("key-b")
        
        # Primary nodes might be different
        primary1 = result1["top_3_nodes"][0]["node_id"]
        primary2 = result2["top_3_nodes"][0]["node_id"]
        
        # Not guaranteed, but very likely with 5 nodes
        # We just verify both return valid results
        assert len(result1["top_3_nodes"]) == 3
        assert len(result2["top_3_nodes"]) == 3


class TestGetTop3NodesForKey:
    """Tests for the get_top3_nodes_for_key function."""
    
    def test_returns_list_of_nodeinfo(self):
        """Should return list of NodeInfo objects."""
        router = create_5_server_cluster()
        nodes = get_top3_nodes_for_key("test-key", router)
        
        assert len(nodes) == 3
        for node in nodes:
            assert isinstance(node, NodeInfo)
    
    def test_all_nodes_unique(self):
        """All returned nodes should be unique."""
        router = create_5_server_cluster()
        nodes = get_top3_nodes_for_key("test-key", router)
        
        node_ids = [n.node_id for n in nodes]
        assert len(node_ids) == len(set(node_ids))


class TestConsistentHashingProperties:
    """Property-based tests for consistent hashing."""
    
    @given(st.sets(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
    def test_all_keys_map_to_valid_node(self, keys):
        """All keys should map to a valid node."""
        ring = ConsistentHashRing()
        for i in range(5):
            ring.add_node(NodeInfo(f"server-{i}", "127.0.0.1", 8000 + i))
        
        for key in keys:
            node = ring.get_node(key)
            assert node is not None
            assert node.node_id.startswith("server-")
    
    @given(st.text(min_size=1), st.integers(min_value=1, max_value=5))
    def test_replication_count_respected(self, key, n):
        """Should return exactly N nodes when possible."""
        ring = ConsistentHashRing()
        for i in range(5):
            ring.add_node(NodeInfo(f"server-{i}", "127.0.0.1", 8000 + i))
        
        nodes = ring.get_nodes(key, n=n)
        
        # Should return min(n, total_nodes)
        assert len(nodes) == min(n, 5)
        
        # All unique
        assert len(set(n.node_id for n in nodes)) == len(nodes)
