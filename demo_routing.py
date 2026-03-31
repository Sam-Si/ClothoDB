#!/usr/bin/env python3
"""
Demo script for ClothoDB consistent hashing and routing.

This script demonstrates:
1. Creating a 5-server cluster
2. Routing keys to the correct nodes
3. Getting top 3 nodes for any key
4. Key distribution statistics
"""

from clotho.consistent_hash import create_5_server_cluster, NodeInfo
from clotho.server import get_top3_nodes_for_key, get_top3_nodes_api


def demo_basic_routing():
    """Demonstrate basic key routing."""
    print("=" * 60)
    print("Demo: Basic Key Routing with 5 Servers")
    print("=" * 60)
    
    # Create a 5-server cluster
    router = create_5_server_cluster(base_port=8000)
    
    print("\nCluster Configuration:")
    print("-" * 40)
    for server in router.get_all_servers():
        print(f"  {server.node_id}: {server.address}")
    
    # Route some keys
    test_keys = [
        "user:123",
        "product:456",
        "order:789",
        "session:abc",
        "cart:xyz"
    ]
    
    print("\nKey Routing Examples:")
    print("-" * 40)
    for key in test_keys:
        primary = router.get_primary_node(key)
        replicas = router.get_replica_nodes(key)
        
        print(f"\nKey: '{key}'")
        print(f"  Primary: {primary.node_id} ({primary.address})")
        print(f"  Replicas: {[n.node_id for n in replicas]}")


def demo_top3_api():
    """Demonstrate the top 3 nodes API."""
    print("\n" + "=" * 60)
    print("Demo: Top 3 Nodes API")
    print("=" * 60)
    
    # Use the convenience API
    test_keys = ["key-a", "key-b", "key-c", "key-d", "key-e"]
    
    for key in test_keys:
        result = get_top3_nodes_api(key)
        
        print(f"\nKey: '{result['key']}'")
        print(f"Total nodes in cluster: {result['total_nodes_in_cluster']}")
        print(f"Replication factor: {result['replication_factor']}")
        print("Top 3 nodes:")
        
        for node in result['top_3_nodes']:
            role = "PRIMARY" if node['is_primary'] else "replica"
            print(f"  {node['rank']}. {node['node_id']} @ {node['address']} ({role})")


def demo_distribution_stats():
    """Demonstrate key distribution statistics."""
    print("\n" + "=" * 60)
    print("Demo: Key Distribution Statistics")
    print("=" * 60)
    
    router = create_5_server_cluster(base_port=8000)
    
    # Get distribution stats
    stats = router.get_distribution_stats()
    
    print("\nVirtual Node Distribution:")
    print("-" * 40)
    print(f"Total physical nodes: {stats['total_nodes']}")
    print(f"Total virtual nodes: {stats['virtual_nodes']}")
    print(f"Avg vnodes per node: {stats['avg_vnodes_per_node']:.1f}")
    print(f"Min vnodes: {stats['min_vnodes']}")
    print(f"Max vnodes: {stats['max_vnodes']}")
    
    print("\nPer-node distribution:")
    for node_id, count in sorted(stats['distribution'].items()):
        bar = "█" * (count // 5)
        print(f"  {node_id}: {count:3d} {bar}")
    
    # Simulate key distribution
    print("\nSimulated Key Distribution (1000 keys):")
    print("-" * 40)
    
    key_counts = {}
    for i in range(1000):
        key = f"key-{i}"
        primary = router.get_primary_node(key)
        key_counts[primary.node_id] = key_counts.get(primary.node_id, 0) + 1
    
    for node_id, count in sorted(key_counts.items()):
        percentage = (count / 1000) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {node_id}: {count:4d} ({percentage:5.1f}%) {bar}")


def demo_node_removal():
    """Demonstrate minimal reorganization on node removal."""
    print("\n" + "=" * 60)
    print("Demo: Node Removal (Minimal Reorganization)")
    print("=" * 60)
    
    router = create_5_server_cluster(base_port=8000)
    
    # Map 100 keys before removal
    keys = [f"key-{i}" for i in range(100)]
    original_mapping = {
        key: router.get_primary_node(key).node_id 
        for key in keys
    }
    
    print("\nOriginal mapping (first 10 keys):")
    for key in keys[:10]:
        print(f"  {key} -> {original_mapping[key]}")
    
    # Remove a node
    print(f"\nRemoving server-3...")
    router.remove_server("server-3")
    
    # Map keys after removal
    new_mapping = {
        key: router.get_primary_node(key).node_id 
        for key in keys
    }
    
    # Count changes
    changed = sum(
        1 for key in keys 
        if original_mapping[key] != new_mapping[key]
    )
    
    print(f"\nKeys that changed node: {changed}/100 ({changed}%)")
    print("(Ideally ~20% - only the keys originally on server-3)")
    
    print("\nNew mapping (first 10 keys):")
    for key in keys[:10]:
        old = original_mapping[key]
        new = new_mapping[key]
        indicator = " ✓" if old == new else " ✗ (changed)"
        print(f"  {key} -> {new}{indicator}")


def demo_api_usage():
    """Demonstrate programmatic API usage."""
    print("\n" + "=" * 60)
    print("Demo: Programmatic API Usage")
    print("=" * 60)
    
    # Create router
    router = create_5_server_cluster(base_port=8000)
    
    print("\nExample: Routing a key to get top 3 nodes")
    print("-" * 40)
    print("Code:")
    print("  from clotho.server import get_top3_nodes_for_key")
    print("  from clotho.consistent_hash import create_5_server_cluster")
    print("")
    print("  router = create_5_server_cluster()")
    print("  nodes = get_top3_nodes_for_key('my-key', router)")
    print("")
    print("Result:")
    
    nodes = get_top3_nodes_for_key("my-key", router)
    for i, node in enumerate(nodes, 1):
        role = "PRIMARY" if i == 1 else f"Replica {i-1}"
        print(f"  {i}. {node.node_id} @ {node.address} ({role})")


def main():
    """Run all demos."""
    demo_basic_routing()
    demo_top3_api()
    demo_distribution_stats()
    demo_node_removal()
    demo_api_usage()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nTo start the actual HTTP servers, run:")
    print("  python -c \"from clotho.server import Cluster; c = Cluster(); c.print_routing_table()\"")
    print("\nThen access the API at:")
    print("  http://127.0.0.1:8000/top3/{key}")
    print("  http://127.0.0.1:8000/route/{key}")
    print("  http://127.0.0.1:8000/status")


if __name__ == "__main__":
    main()
