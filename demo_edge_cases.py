#!/usr/bin/env python3
"""
Demo: Edge Case Handling in ClothoDB

This demo showcases how ClothoDB handles various distributed systems edge cases:
1. Hinted Handoff - Temporary node failures
2. Read Repair - Inconsistent replicas
3. Circuit Breakers - Cascading failure prevention
4. Sloppy Quorum - Network partitions
5. Vector Clock Pruning - Unbounded clock growth

Run with: python demo_edge_cases.py
"""

import asyncio
import time
from clotho import (
    VectorClock,
    create_5_server_cluster,
    HintedHandoffManager,
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    QuorumManager,
    QuorumType,
    VectorClockPruner,
    ResilientStorage,
    ReadRepairHandler,
)


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title):
    """Print a section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


async def demo_hinted_handoff():
    """Demo: Hinted Handoff for temporary node failures."""
    print_header("DEMO 1: Hinted Handoff (Temporary Node Failures)")
    
    print("""
Scenario: A write targets 3 replicas (N=3), but one node is temporarily down.
Solution: Store a "hint" on an available node and forward when target recovers.

This is how DynamoDB handles temporary failures without rejecting writes.
""")
    
    # Create manager for node-1
    manager = HintedHandoffManager("node-1")
    
    print_section("Storing Hints")
    
    # Simulate failed writes to different nodes
    clock = VectorClock.new("client-1").increment("client-1")
    
    manager.store_hint("node-3", "user:123", {"name": "Alice"}, clock)
    print(f"✓ Stored hint for node-3, key='user:123'")
    
    manager.store_hint("node-3", "user:456", {"name": "Bob"}, clock)
    print(f"✓ Stored hint for node-3, key='user:456'")
    
    manager.store_hint("node-5", "product:789", {"name": "Widget"}, clock)
    print(f"✓ Stored hint for node-5, key='product:789'")
    
    print_section("Hint Statistics")
    stats = manager.get_stats()
    print(f"Total pending hints: {stats['total_pending_hints']}")
    print(f"Nodes with hints: {stats['nodes_with_hints']}")
    for node_id, count in stats['hints_by_node'].items():
        print(f"  - {node_id}: {count} hints")
    
    print_section("Expired Hint Cleanup")
    # Create an expired hint manually
    from clotho.resilience import Hint
    expired_hint = Hint(
        target_node_id="node-2",
        key="old-key",
        value="old-value",
        vector_clock=clock,
        timestamp=time.time() - 7200  # 2 hours ago
    )
    manager.hints["node-2"].append(expired_hint)
    print(f"✓ Added expired hint for node-2")
    
    cleaned = manager.cleanup_expired_hints()
    print(f"✓ Cleaned up {cleaned} expired hints")
    
    print("\n✅ Hinted Handoff ensures writes succeed even when nodes are down!")


async def demo_circuit_breaker():
    """Demo: Circuit Breaker pattern."""
    print_header("DEMO 2: Circuit Breaker (Cascading Failure Prevention)")
    
    print("""
Scenario: A node is failing repeatedly. Without intervention, all requests
          to that node would fail, causing cascading failures.
Solution: Open the circuit after threshold failures, reject requests immediately.

This prevents wasting resources on failing nodes and allows them to recover.
""")
    
    cb = CircuitBreaker(
        "unreliable-node",
        failure_threshold=3,
        recovery_timeout=1.0,
        half_open_max_calls=2
    )
    
    print_section("Normal Operation (Circuit Closed)")
    
    async def success_op():
        return "success"
    
    result = await cb.call(success_op)
    print(f"✓ Request succeeded: {result}")
    print(f"  Circuit state: {cb.get_state().value}")
    
    print_section("Failures Accumulate")
    
    async def fail_op():
        raise ConnectionError("Node is down!")
    
    for i in range(3):
        try:
            await cb.call(fail_op)
        except ConnectionError:
            print(f"✗ Request {i+1} failed (failure_count={cb.failure_count})")
    
    print(f"\n  Circuit state: {cb.get_state().value} (threshold reached)")
    
    print_section("Circuit Open - Fast Fail")
    
    try:
        await cb.call(success_op)  # Even success op will fail
    except Exception as e:
        print(f"✗ Request rejected immediately: {type(e).__name__}")
        print(f"  No resources wasted on failing node!")
    
    print_section("Recovery (Half-Open)")
    print("Waiting for recovery timeout...")
    await asyncio.sleep(1.1)
    
    # Circuit should transition to half-open
    result = await cb.call(success_op)
    print(f"✓ Test request succeeded: {result}")
    print(f"  Circuit state: {cb.get_state().value}")
    
    print("\n✅ Circuit Breaker prevents cascading failures!")


async def demo_retry_policy():
    """Demo: Retry policy with exponential backoff."""
    print_header("DEMO 3: Retry Policy (Exponential Backoff)")
    
    print("""
Scenario: Transient failures (network blips) should be retried, but we need
to avoid the "thundering herd" problem where all clients retry simultaneously.
Solution: Exponential backoff with jitter.
""")
    
    print_section("Exponential Backoff Delays")
    
    policy = RetryPolicy(
        max_retries=5,
        base_delay=0.1,  # 100ms
        max_delay=2.0,   # 2s cap
        exponential_base=2.0,
        jitter=False
    )
    
    print("Delay pattern (no jitter):")
    for attempt in range(6):
        delay = policy.get_delay(attempt)
        print(f"  Attempt {attempt}: {delay*1000:.0f}ms")
    
    print_section("With Jitter (Prevents Thundering Herd)")
    
    policy_jitter = RetryPolicy(
        base_delay=1.0,
        jitter=True
    )
    
    delays = [policy_jitter.get_delay(0) for _ in range(10)]
    print("10 delays for attempt 0 (base=1.0s, jitter enabled):")
    for i, d in enumerate(delays):
        print(f"  Delay {i+1}: {d:.2f}s ({d*100:.0f}% of base)")
    
    print("\n✅ Jitter spreads out retry attempts, preventing system overload!")


async def demo_sloppy_quorum():
    """Demo: Sloppy quorum for network partitions."""
    print_header("DEMO 4: Sloppy Quorum (Network Partitions)")
    
    print("""
Scenario: Network partition separates the cluster. Standard quorum would
          reject all writes since designated replicas are unreachable.
Solution: Sloppy quorum - write to ANY available W nodes, not just designated.

This sacrifices strict consistency for availability during partitions.
""")
    
    router = create_5_server_cluster()
    
    print_section("Strict Quorum (Normal Case)")
    
    strict_qm = QuorumManager(router, n=3, w=2, r=2, quorum_type=QuorumType.STRICT)
    
    # All nodes available
    all_nodes = {f"server-{i}" for i in range(1, 6)}
    write_nodes = strict_qm.get_write_nodes("user:123", all_nodes)
    
    print(f"Available nodes: {len(all_nodes)}")
    print(f"Write nodes selected: {[n.node_id for n in write_nodes]}")
    print(f"Write quorum possible: {strict_qm.is_write_quorum_possible(all_nodes)}")
    
    print_section("Strict Quorum (Partitioned - Fails)")
    
    # Partition: only 1 designated node available
    preference = router.get_replica_nodes("user:123")
    partitioned = {preference[0].node_id}  # Only primary available
    
    write_nodes = strict_qm.get_write_nodes("user:123", partitioned)
    print(f"Available nodes: {len(partitioned)}")
    print(f"Write nodes selected: {[n.node_id for n in write_nodes]}")
    print(f"Write quorum possible: {strict_qm.is_write_quorum_possible(partitioned)}")
    print(f"  ✗ Would reject writes!")
    
    print_section("Sloppy Quorum (Partitioned - Succeeds)")
    
    sloppy_qm = QuorumManager(router, n=3, w=2, r=2, quorum_type=QuorumType.SLOPPY)
    
    # Add some non-preference nodes to available set
    partitioned = {preference[0].node_id, "server-4", "server-5"}
    
    write_nodes = sloppy_qm.get_write_nodes("user:123", partitioned)
    print(f"Available nodes: {len(partitioned)}")
    print(f"Write nodes selected: {[n.node_id for n in write_nodes]}")
    print(f"  ✓ Writes can proceed!")
    print(f"\n  Note: Vector clocks will detect conflicts when partition heals")
    
    print("\n✅ Sloppy Quorum maintains availability during partitions!")


async def demo_vector_clock_pruning():
    """Demo: Vector clock pruning."""
    print_header("DEMO 5: Vector Clock Pruning (Unbounded Growth)")
    
    print("""
Scenario: In a large cluster, vector clocks can grow with every node that
          touches a value, leading to unbounded memory usage.
Solution: Prune clocks to MAX entries (like DynamoDB's 10-node limit).
""")
    
    pruner = VectorClockPruner()
    
    print_section("Small Clock (No Pruning Needed)")
    
    small_clock = VectorClock({"a": 1, "b": 2, "c": 3})
    print(f"Original clock: {small_clock.to_dict()}")
    print(f"Size: {len(small_clock)} entries")
    print(f"Needs pruning: {pruner.should_prune(small_clock)}")
    
    pruned = pruner.prune(small_clock)
    print(f"After pruning: {pruned.to_dict()}")
    
    print_section("Large Clock (Pruning Required)")
    
    # Simulate clock with 15 nodes
    large_clock = VectorClock({f"node-{i}": i for i in range(15)})
    print(f"Original clock size: {len(large_clock)} entries")
    print(f"Needs pruning: {pruner.should_prune(large_clock)}")
    
    pruned = pruner.prune(large_clock)
    print(f"After pruning: {len(pruned)} entries")
    print(f"Pruned clock: {pruned.to_dict()}")
    
    print(f"\n  Strategy: Keep entries with highest timestamps (most recent)")
    print(f"  Dropped: node-0 through node-4 (oldest)")
    print(f"  Kept: node-5 through node-14 (newest)")
    
    print("\n✅ Pruning prevents unbounded memory growth!")


async def demo_read_repair():
    """Demo: Read repair for consistency."""
    print_header("DEMO 6: Read Repair (Consistency)")
    
    print("""
Scenario: Replicas diverge due to network issues or concurrent writes.
Solution: During reads, compare versions from R replicas and repair divergent ones.

This is "lazy" consistency - we fix inconsistencies when they're detected.
""")
    
    router = create_5_server_cluster()
    handler = ReadRepairHandler(router)
    
    print_section("No Conflict (Versions Agree)")
    
    # Sequential updates - no conflict
    clock1 = VectorClock.new("a").increment("a")  # [a:1]
    clock2 = clock1.increment("b")  # [a:1, b:1] - b happens after a
    
    versions = [
        ("value-v1", clock1, "node-1"),
        ("value-v2", clock2, "node-2"),
    ]
    
    has_conflict = handler._check_for_conflicts(versions)
    print(f"Clock 1: {clock1.to_dict()}")
    print(f"Clock 2: {clock2.to_dict()}")
    print(f"Clock 1 happens-before Clock 2: {clock1.compare(clock2).name}")
    print(f"Conflict detected: {has_conflict}")
    print(f"  ✓ No repair needed, return latest version")
    
    print_section("Conflict Detected (Concurrent Writes)")
    
    # Concurrent writes - conflict
    clock1 = VectorClock.new("a").increment("a")  # [a:1]
    clock2 = VectorClock.new("b").increment("b")  # [b:1]
    
    versions = [
        ("value-from-a", clock1, "node-1"),
        ("value-from-b", clock2, "node-2"),
    ]
    
    has_conflict = handler._check_for_conflicts(versions)
    print(f"Clock 1: {clock1.to_dict()}")
    print(f"Clock 2: {clock2.to_dict()}")
    print(f"Clock 1 vs Clock 2: {clock1.compare(clock2).name}")
    print(f"Conflict detected: {has_conflict}")
    
    print_section("Reconciliation")
    
    merged_val, merged_clock = handler._reconcile_versions(versions)
    print(f"Merged clock: {merged_clock.to_dict()}")
    print(f"Merged value type: {type(merged_val).__name__}")
    
    if isinstance(merged_val, dict) and merged_val.get("conflict"):
        print(f"  ⚠ Siblings returned for client-side resolution")
        print(f"  Values: {merged_val['values']}")
        print(f"\n  Next: Async write repair to all replicas")
    
    print("\n✅ Read Repair fixes inconsistencies lazily!")


async def main():
    """Run all demos."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           ClothoDB: Edge Case Handling Demonstrations                ║
║                                                                      ║
║     How DynamoDB handles pesky distributed systems edge cases       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    await demo_hinted_handoff()
    await demo_circuit_breaker()
    await demo_retry_policy()
    await demo_sloppy_quorum()
    await demo_vector_clock_pruning()
    await demo_read_repair()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                         Summary                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Edge Case              | Solution                                   ║
╠═════════════════════════╪════════════════════════════════════════════╣
║  Temporary Node Failure | Hinted Handoff                             ║
║  Cascading Failures     | Circuit Breaker                            ║
║  Transient Errors       | Exponential Backoff + Jitter               ║
║  Network Partitions     | Sloppy Quorum                              ║
║  Clock Growth           | Vector Clock Pruning (10-node limit)       ║
║  Inconsistent Replicas  | Read Repair                                ║
╚══════════════════════════════════════════════════════════════════════╝

All edge cases handled! ✅
""")


if __name__ == "__main__":
    asyncio.run(main())
