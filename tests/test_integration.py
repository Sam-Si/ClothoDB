"""
Integration tests for multi-node distributed scenarios.

These tests simulate real distributed system scenarios:
1. Two nodes communicating
2. Network partitions and healing
3. Concurrent writes and conflict detection
4. Causal consistency verification
"""

import pytest
from typing import Set

from clotho.vector_clock import VectorClock, CausalityRelation
from clotho.node import Node, Cluster


class TestTwoNodeCommunication:
    """Tests for basic two-node communication scenarios."""
    
    def test_initial_clocks_are_independent(self):
        """
        Two nodes that haven't communicated have concurrent clocks.
        
        Scenario:
        - Node A writes x=1
        - Node B writes y=2
        - No communication between them
        
        Expected: Their clocks are concurrent (neither precedes the other)
        """
        node_a = Node("A")
        node_b = Node("B")
        
        node_a.write("x", 1)
        node_b.write("y", 2)
        
        # Clocks should be concurrent
        assert node_a.vector_clock.is_concurrent_with(node_b.vector_clock)
    
    def test_send_message_establishes_causality(self):
        """
        When A sends a message to B, A's events causally precede B's receive.
        
        Scenario:
        1. Node A writes x=1 (clock: A:1)
        2. Node A sends message to Node B
        3. Node B receives and increments (clock: A:1, B:1)
        
        Expected: A's write happens before B's receive
        """
        node_a = Node("A")
        node_b = Node("B")
        
        # A writes
        event_a = node_a.write("x", 1)
        clock_after_write = node_a.vector_clock.copy()
        
        # A sends to B
        node_a.send_message(node_b, "x")
        
        # B's clock should happen after A's write
        assert node_b.vector_clock.happens_after(clock_after_write)
        # Specifically, B should know about A's write
        assert node_b.vector_clock.get_timestamp("A") >= 1
    
    def test_causality_chain(self):
        """
        Causality chains: A -> B -> C means A -> C.
        
        Scenario:
        1. A writes x=1
        2. A sends to B
        3. B sends to C
        
        Expected: A's write causally precedes C's state
        """
        node_a = Node("A")
        node_b = Node("B")
        node_c = Node("C")
        
        # Chain of events
        event_a = node_a.write("x", 1)
        node_a.send_message(node_b, "x")
        node_b.send_message(node_c, "x")
        
        # C should know about A's event
        assert node_c.vector_clock.happens_after(event_a.vector_clock)
        assert node_c.vector_clock.get_timestamp("A") >= 1
        assert node_c.vector_clock.get_timestamp("B") >= 1
    
    def test_bidirectional_communication(self):
        """
        Bidirectional communication eventually synchronizes clocks.
        
        Scenario:
        1. A writes x=1
        2. B writes y=2
        3. A sends to B
        4. B sends to A
        
        Expected: Both nodes have the same (merged) clock
        """
        node_a = Node("A")
        node_b = Node("B")
        
        node_a.write("x", 1)
        node_b.write("y", 2)
        
        # Bidirectional sync
        node_a.send_message(node_b, "x")
        node_b.send_message(node_a, "y")
        
        # Both should know about both writes
        assert node_a.vector_clock.get_timestamp("A") >= 1
        assert node_a.vector_clock.get_timestamp("B") >= 1
        assert node_b.vector_clock.get_timestamp("A") >= 1
        assert node_b.vector_clock.get_timestamp("B") >= 1


class TestNetworkPartition:
    """Tests for network partition scenarios."""
    
    def test_partition_creates_concurrent_events(self):
        """
        During a partition, writes on different sides are concurrent.
        
        Scenario:
        1. Nodes A, B, C start connected
        2. Partition: {A} | {B, C}
        3. A writes x=1
        4. B writes x=2
        5. Partition heals
        
        Expected: The two writes are concurrent (conflict!)
        """
        cluster = Cluster()
        node_a = cluster.add_node("A")
        node_b = cluster.add_node("B")
        node_c = cluster.add_node("C")
        
        # Initial sync
        node_a.send_message(node_b, "init")
        node_b.send_message(node_c, "init")
        
        # Simulate partition: A can't talk to B or C
        # During partition:
        event_a = node_a.write("x", 1)  # A writes x=1
        event_b = node_b.write("x", 2)  # B writes x=2
        
        # The writes should be concurrent
        assert event_a.vector_clock.is_concurrent_with(event_b.vector_clock)
    
    def test_partition_healing_detects_conflicts(self):
        """
        When partition heals, concurrent writes are detected as conflicts.
        """
        cluster = Cluster()
        node_a = cluster.add_node("A")
        node_b = cluster.add_node("B")
        
        # Initial sync
        node_a.send_message(node_b, "init")
        
        # Partition
        event_a = node_a.write("x", 1)
        event_b = node_b.write("x", 2)
        
        # Partition heals - A sends to B
        node_a.send_message(node_b, "x")
        
        # B should detect the conflict
        conflicts = cluster.find_concurrent_conflicts()
        assert len(conflicts) >= 1
        
        # Verify the conflict pair
        conflict_found = False
        for e1, e2 in conflicts:
            if (e1.key == "x" and e2.key == "x" and
                e1.node_id != e2.node_id):
                conflict_found = True
                break
        assert conflict_found


class TestCausalConsistency:
    """Tests verifying causal consistency guarantees."""
    
    def test_read_your_writes(self):
        """
        Read-Your-Writes: A node should see its own writes.
        
        This is a fundamental consistency guarantee.
        """
        node = Node("A")
        
        node.write("x", 1)
        result = node.read("x")
        
        assert result is not None
        assert result[0] == 1
    
    def test_monotonic_reads(self):
        """
        Monotonic Reads: If a node reads a value, subsequent reads
        should see the same or newer value.
        
        Note: In this simplified implementation, send_message transfers the
        vector clock but not the actual data. For causality tracking, we
        verify that the recipient's clock advances causally.
        """
        node_a = Node("A")
        node_b = Node("B")
        
        # A writes x=1
        event1 = node_a.write("x", 1)
        
        # A sends to B - B's clock advances
        node_a.send_message(node_b, "x")
        
        # B's clock should now causally follow A's write
        assert node_b.vector_clock.happens_after(event1.vector_clock)
        
        # Record B's clock after first sync
        first_sync_clock = node_b.vector_clock.copy()
        
        # A writes x=2 and sends to B
        event2 = node_a.write("x", 2)
        node_a.send_message(node_b, "x")
        
        # B's new clock should causally follow both the first sync and A's new write
        assert node_b.vector_clock.happens_after(first_sync_clock)
        assert node_b.vector_clock.happens_after(event2.vector_clock)
    
    def test_writes_follow_reads(self):
        """
        Writes-Follow-Reads: If a node reads value v1 written by node A,
        then writes v2, then v2 causally depends on v1.
        """
        node_a = Node("A")
        node_b = Node("B")
        
        # A writes x=1
        event_v1 = node_a.write("x", 1)
        node_a.send_message(node_b, "x")
        
        # B reads x=1
        node_b.read("x")
        
        # B writes y=2 (should causally depend on reading x=1)
        event_v2 = node_b.write("y", 2)
        
        # v2 should happen after v1
        assert event_v2.vector_clock.happens_after(event_v1.vector_clock)
    
    def test_causal_delivery(self):
        """
        Causal Delivery: If event A causally precedes event B,
        all nodes should deliver A before B.
        """
        node_a = Node("A")
        node_b = Node("B")
        node_c = Node("C")
        
        # A writes x=1
        event_a = node_a.write("x", 1)
        
        # A sends to B
        node_a.send_message(node_b, "x")
        
        # B writes y=2 (causally depends on A's write)
        event_b = node_b.write("y", 2)
        
        # B sends to C
        node_b.send_message(node_c, "y")
        
        # C should know about both events in causal order
        assert node_c.is_causally_ready(event_a)
        assert node_c.is_causally_ready(event_b)
        
        # C's clock should be after both
        assert node_c.vector_clock.happens_after(event_a.vector_clock)
        assert node_c.vector_clock.happens_after(event_b.vector_clock)


class TestEventOrdering:
    """Tests for event ordering and history."""
    
    def test_local_events_are_ordered(self):
        """
        On a single node, events should be totally ordered.
        """
        node = Node("A")
        
        event1 = node.write("x", 1)
        event2 = node.write("x", 2)
        event3 = node.write("y", 3)
        
        # Later events should happen after earlier events
        assert event2.vector_clock.happens_after(event1.vector_clock)
        assert event3.vector_clock.happens_after(event2.vector_clock)
        assert event3.vector_clock.happens_after(event1.vector_clock)
    
    def test_causal_history_is_sorted(self):
        """
        get_causal_history should return events in causal order.
        """
        node = Node("A")
        
        events = []
        for i in range(5):
            events.append(node.write(f"key{i}", i))
        
        history = node.get_causal_history()
        
        # Each event should happen after all previous events
        for i in range(1, len(history)):
            prev_clock = history[i-1].vector_clock
            curr_clock = history[i].vector_clock
            # Note: concurrent events might exist, but for single node,
            # they should all be ordered
            assert curr_clock.happens_after(prev_clock)
    
    def test_concurrent_events_detected(self):
        """
        Concurrent events on different nodes should be detected.
        """
        node_a = Node("A")
        node_b = Node("B")
        
        # Concurrent writes
        event_a = node_a.write("x", 1)
        event_b = node_b.write("y", 2)
        
        # They should be concurrent
        assert event_a.vector_clock.is_concurrent_with(event_b.vector_clock)
        
        # After communication, they're no longer concurrent
        node_a.send_message(node_b, "x")
        # B's new clock dominates both
        assert node_b.vector_clock.happens_after(event_a.vector_clock)
        assert node_b.vector_clock.happens_after(event_b.vector_clock)


class TestConflictDetection:
    """Tests for detecting and handling conflicts."""
    
    def test_concurrent_writes_same_key_are_conflicts(self):
        """
        Two concurrent writes to the same key are conflicts.
        """
        cluster = Cluster()
        node_a = cluster.add_node("A")
        node_b = cluster.add_node("B")
        
        # Concurrent writes to same key
        event_a = node_a.write("x", 1)
        event_b = node_b.write("x", 2)
        
        # They should be concurrent
        assert event_a.vector_clock.is_concurrent_with(event_b.vector_clock)
        
        # This is a conflict
        conflicts = cluster.find_concurrent_conflicts()
        assert len(conflicts) == 1
        
        conflict_pair = conflicts[0]
        assert conflict_pair[0].key == "x"
        assert conflict_pair[1].key == "x"
    
    def test_sequential_writes_not_conflicts(self):
        """
        Sequential writes (with causality) are not conflicts.
        """
        cluster = Cluster()
        node_a = cluster.add_node("A")
        node_b = cluster.add_node("B")
        
        # Sequential writes (A writes, sends to B, B writes)
        node_a.write("x", 1)
        node_a.send_message(node_b, "x")
        node_b.write("x", 2)
        
        # No conflicts - B's write causally follows A's
        conflicts = cluster.find_concurrent_conflicts()
        assert len(conflicts) == 0
    
    def test_multiway_conflict(self):
        """
        Multiple concurrent writes from different nodes are all conflicts.
        """
        cluster = Cluster()
        node_a = cluster.add_node("A")
        node_b = cluster.add_node("B")
        node_c = cluster.add_node("C")
        
        # Three concurrent writes
        node_a.write("x", 1)
        node_b.write("x", 2)
        node_c.write("x", 3)
        
        # All pairs are conflicts
        conflicts = cluster.find_concurrent_conflicts()
        # Should have 3 pairs: (A,B), (A,C), (B,C)
        assert len(conflicts) == 3


class TestClockPropagation:
    """Tests for vector clock propagation across the cluster."""
    
    def test_clock_grows_with_communication(self):
        """
        As nodes communicate, their clocks grow to include all nodes.
        """
        cluster = Cluster()
        nodes = [cluster.add_node(f"N{i}") for i in range(5)]
        
        # Chain of communication
        for i in range(len(nodes) - 1):
            nodes[i].write(f"key{i}", i)
            nodes[i].send_message(nodes[i + 1], f"key{i}")
        
        # Last node should know about all previous nodes
        last_node = nodes[-1]
        for i in range(len(nodes)):
            assert f"N{i}" in last_node.known_nodes
    
    def test_gossip_protocol_simulation(self):
        """
        Simulate gossip protocol - eventually all nodes converge.
        """
        cluster = Cluster()
        node_a = cluster.add_node("A")
        node_b = cluster.add_node("B")
        node_c = cluster.add_node("C")
        
        # Each node writes
        node_a.write("x", 1)
        node_b.write("y", 2)
        node_c.write("z", 3)
        
        # Gossip rounds
        node_a.send_message(node_b, "x")
        node_b.send_message(node_c, "y")
        node_c.send_message(node_a, "z")
        
        # Another round
        node_a.send_message(node_b, "z")
        node_b.send_message(node_c, "x")
        
        # All nodes should know about all writes
        for node in [node_a, node_b, node_c]:
            assert node.vector_clock.get_timestamp("A") >= 1
            assert node.vector_clock.get_timestamp("B") >= 1
            assert node.vector_clock.get_timestamp("C") >= 1


class TestEdgeCases:
    """Edge cases and boundary conditions."""
    
    def test_single_node_cluster(self):
        """
        A single node should work correctly.
        """
        node = Node("solo")
        
        for i in range(100):
            node.write(f"key{i}", i)
        
        assert len(node.events) == 100
        assert node.vector_clock.get_timestamp("solo") == 100
    
    def test_many_nodes(self):
        """
        System should handle many nodes.
        """
        cluster = Cluster()
        num_nodes = 50
        
        nodes = [cluster.add_node(f"N{i}") for i in range(num_nodes)]
        
        # Each node writes
        for node in nodes:
            node.write("x", 1)
        
        # Central node collects all
        central = nodes[0]
        for node in nodes[1:]:
            node.send_message(central, "x")
        
        # Central should know about all nodes
        for i in range(num_nodes):
            assert central.vector_clock.get_timestamp(f"N{i}") >= 1
    
    def test_rapid_writes(self):
        """
        Rapid writes on a single node.
        """
        node = Node("A")
        
        events = []
        for i in range(1000):
            events.append(node.write("x", i))
        
        # All events should be ordered
        for i in range(1, len(events)):
            assert events[i].vector_clock.happens_after(events[i-1].vector_clock)
    
    def test_empty_send(self):
        """
        Sending a message for a key that doesn't exist.
        
        The sender's clock is propagated to the recipient, who increments
        their own counter upon receipt.
        """
        node_a = Node("A")
        node_b = Node("B")
        
        # Send without writing first
        node_a.send_message(node_b, "nonexistent")
        
        # Recipient (B) should have incremented their counter
        assert node_b.vector_clock.get_timestamp("B") >= 1
        # Recipient knows about sender (A) at counter 0 (initial state)
        assert node_b.vector_clock.get_timestamp("A") == 0


class TestVectorClockInvariants:
    """
    Invariants that must always hold in a distributed system.
    
    These are properties that should be true regardless of the
    specific sequence of operations.
    """
    
    def test_own_counter_never_decreases(self):
        """
        A node's own counter should never decrease.
        """
        node = Node("A")
        
        prev_counter = node.vector_clock.get_timestamp("A")
        
        for i in range(100):
            if i % 3 == 0:
                node.write("x", i)
            elif i % 3 == 1:
                other = Node(f"temp_{i}")
                other.send_message(node, "y")
            else:
                node.read("x")
            
            curr_counter = node.vector_clock.get_timestamp("A")
            assert curr_counter >= prev_counter
            prev_counter = curr_counter
    
    def test_merge_result_dominates_both(self):
        """
        merge(A, B) should dominate both A and B.
        """
        for _ in range(100):
            # Random clocks
            import random
            clock_a = VectorClock.from_dict({
                "a": random.randint(0, 10),
                "b": random.randint(0, 10),
            })
            clock_b = VectorClock.from_dict({
                "b": random.randint(0, 10),
                "c": random.randint(0, 10),
            })
            
            merged = clock_a.merge(clock_b)
            
            # merged should dominate both
            assert merged.happens_after(clock_a) or merged.is_equal(clock_a)
            assert merged.happens_after(clock_b) or merged.is_equal(clock_b)
    
    def test_causality_is_transitive_in_cluster(self):
        """
        If A -> B and B -> C in a cluster, then A -> C.
        """
        cluster = Cluster()
        node_a = cluster.add_node("A")
        node_b = cluster.add_node("B")
        node_c = cluster.add_node("C")
        
        # A -> B
        event_a = node_a.write("x", 1)
        node_a.send_message(node_b, "x")
        
        # B -> C
        node_b.write("y", 2)  # B's own event
        node_b.send_message(node_c, "y")
        event_c = node_c.write("z", 3)
        
        # A -> C
        assert event_c.vector_clock.happens_after(event_a.vector_clock)
