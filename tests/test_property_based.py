"""
Property-based tests (fuzzing) for causality tracking.

These tests use Hypothesis to generate random scenarios and verify
that the vector clock implementation satisfies mathematical properties.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, event, note

from clotho.vector_clock import VectorClock, CausalityRelation
from clotho.node import Node, Cluster


# Strategies for generating test data
node_ids = st.sampled_from(["A", "B", "C", "D", "E", "F", "G", "H"])

@st.composite
def vector_clock_dicts(draw):
    """Generate random vector clock dictionaries."""
    nodes = draw(st.sets(node_ids, min_size=1, max_size=5))
    return {
        node: draw(st.integers(min_value=0, max_value=100))
        for node in nodes
    }


class TestVectorClockProperties:
    """
    Mathematical properties that must hold for all vector clocks.
    
    These are universal truths about vector clocks that should hold
    regardless of the specific values.
    """
    
    @given(vector_clock_dicts(), vector_clock_dicts())
    def test_merge_is_commutative(self, dict_a, dict_b):
        """
        For all clocks A, B: A.merge(B) == B.merge(A)
        
        This is a fundamental property: the order of merging shouldn't matter.
        """
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        
        merge_ab = clock_a.merge(clock_b)
        merge_ba = clock_b.merge(clock_a)
        
        assert merge_ab.is_equal(merge_ba), \
            f"Merge not commutative: {dict_a}.merge({dict_b}) != {dict_b}.merge({dict_a})"
    
    @given(vector_clock_dicts(), vector_clock_dicts(), vector_clock_dicts())
    def test_merge_is_associative(self, dict_a, dict_b, dict_c):
        """
        For all clocks A, B, C: (A.merge(B)).merge(C) == A.merge(B.merge(C))
        
        This allows us to merge clocks in any grouping order.
        """
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        clock_c = VectorClock.from_dict(dict_c)
        
        merge_ab_c = clock_a.merge(clock_b).merge(clock_c)
        merge_a_bc = clock_a.merge(clock_b.merge(clock_c))
        
        assert merge_ab_c.is_equal(merge_a_bc), \
            f"Merge not associative for {dict_a}, {dict_b}, {dict_c}"
    
    @given(vector_clock_dicts())
    def test_merge_with_self_is_idempotent(self, dict_a):
        """
        For all clocks A: A.merge(A) == A
        
        Merging a clock with itself should be a no-op.
        """
        clock_a = VectorClock.from_dict(dict_a)
        merged = clock_a.merge(clock_a)
        
        assert merged.is_equal(clock_a), \
            f"Merge not idempotent for {dict_a}"
    
    @given(vector_clock_dicts())
    def test_comparison_is_reflexive(self, dict_a):
        """
        For all clocks A: A == A
        
        Every clock is equal to itself.
        """
        clock_a = VectorClock.from_dict(dict_a)
        assert clock_a.is_equal(clock_a)
        assert clock_a.compare(clock_a) == CausalityRelation.EQUAL
    
    @given(vector_clock_dicts(), vector_clock_dicts())
    def test_comparison_is_symmetric_for_equality(self, dict_a, dict_b):
        """
        For all clocks A, B: if A == B then B == A
        """
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        
        if clock_a.is_equal(clock_b):
            assert clock_b.is_equal(clock_a), \
                f"Equality not symmetric: {dict_a} == {dict_b} but not vice versa"
    
    @given(vector_clock_dicts(), vector_clock_dicts(), vector_clock_dicts())
    def test_happens_before_is_transitive(self, dict_a, dict_b, dict_c):
        """
        For all clocks A, B, C:
        if A -> B and B -> C, then A -> C
        
        This is the fundamental causality property.
        """
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        clock_c = VectorClock.from_dict(dict_c)
        
        a_before_b = clock_a.happens_before(clock_b)
        b_before_c = clock_b.happens_before(clock_c)
        
        if a_before_b and b_before_c:
            assert clock_a.happens_before(clock_c), \
                f"Causality not transitive: {dict_a} -> {dict_b} -> {dict_c} but not {dict_a} -> {dict_c}"
    
    @given(vector_clock_dicts(), vector_clock_dicts())
    def test_concurrent_is_symmetric(self, dict_a, dict_b):
        """
        For all clocks A, B: if A || B then B || A
        
        Concurrency is a symmetric relationship.
        """
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        
        a_concurrent_b = clock_a.is_concurrent_with(clock_b)
        b_concurrent_a = clock_b.is_concurrent_with(clock_a)
        
        assert a_concurrent_b == b_concurrent_a, \
            f"Concurrency not symmetric: {dict_a} || {dict_b} is {a_concurrent_b} but reverse is {b_concurrent_a}"
    
    @given(vector_clock_dicts(), vector_clock_dicts())
    def test_mutual_exclusivity_of_relations(self, dict_a, dict_b):
        """
        For all clocks A, B, exactly one of these is true:
        - A == B
        - A -> B
        - B -> A
        - A || B
        
        These relations are mutually exclusive and exhaustive.
        """
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        
        relation = clock_a.compare(clock_b)
        
        # Check that helper methods are consistent
        if relation == CausalityRelation.EQUAL:
            assert clock_a.is_equal(clock_b)
            assert not clock_a.happens_before(clock_b)
            assert not clock_b.happens_before(clock_a)
            assert not clock_a.is_concurrent_with(clock_b)
        elif relation == CausalityRelation.HAPPENS_BEFORE:
            assert not clock_a.is_equal(clock_b)
            assert clock_a.happens_before(clock_b)
            assert not clock_b.happens_before(clock_a)
            assert not clock_a.is_concurrent_with(clock_b)
        elif relation == CausalityRelation.HAPPENS_AFTER:
            assert not clock_a.is_equal(clock_b)
            assert not clock_a.happens_before(clock_b)
            assert clock_b.happens_before(clock_a)
            assert not clock_a.is_concurrent_with(clock_b)
        else:  # CONCURRENT
            assert not clock_a.is_equal(clock_b)
            assert not clock_a.happens_before(clock_b)
            assert not clock_b.happens_before(clock_a)
            assert clock_a.is_concurrent_with(clock_b)
    
    @given(vector_clock_dicts(), vector_clock_dicts())
    def test_merge_dominates_both(self, dict_a, dict_b):
        """
        For all clocks A, B:
        - A.merge(B) >= A
        - A.merge(B) >= B
        
        The merge result should dominate both inputs.
        """
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        
        merged = clock_a.merge(clock_b)
        
        # merged should happen after or be equal to both
        rel_a = merged.compare(clock_a)
        rel_b = merged.compare(clock_b)
        
        assert rel_a in (CausalityRelation.HAPPENS_AFTER, CausalityRelation.EQUAL), \
            f"Merge doesn't dominate first argument: merge({dict_a}, {dict_b})"
        assert rel_b in (CausalityRelation.HAPPENS_AFTER, CausalityRelation.EQUAL), \
            f"Merge doesn't dominate second argument: merge({dict_a}, {dict_b})"
    
    @given(vector_clock_dicts(), st.data())
    def test_increment_increases_own_counter(self, dict_dict, data):
        """
        For all clocks C and nodes n:
        C.increment(n)[n] == C[n] + 1
        """
        assume(len(dict_dict) > 0)
        
        clock = VectorClock.from_dict(dict_dict)
        node = data.draw(st.sampled_from(list(dict_dict.keys())))
        
        incremented = clock.increment(node)
        
        assert incremented.get_timestamp(node) == clock.get_timestamp(node) + 1, \
            f"Increment didn't increase counter for {node}"
        
        # All other nodes should be unchanged
        for other_node in dict_dict:
            if other_node != node:
                assert incremented.get_timestamp(other_node) == clock.get_timestamp(other_node), \
                    f"Increment modified wrong node's counter"
    
    @given(vector_clock_dicts(), vector_clock_dicts())
    def test_happens_before_implies_component_wise_leq(self, dict_a, dict_b):
        """
        For all clocks A, B:
        if A -> B, then for all nodes n: A[n] <= B[n]
        
        This is the definition of happens-before.
        """
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        
        if clock_a.happens_before(clock_b):
            all_nodes = set(dict_a.keys()) | set(dict_b.keys())
            for node in all_nodes:
                assert clock_a.get_timestamp(node) <= clock_b.get_timestamp(node), \
                    f"Happens-before violated: {node} has {clock_a.get_timestamp(node)} > {clock_b.get_timestamp(node)}"
    
    @given(vector_clock_dicts(), vector_clock_dicts())
    def test_concurrent_implies_incomparable(self, dict_a, dict_b):
        """
        For all clocks A, B:
        if A || B, then NOT(A <= B) AND NOT(B <= A)
        
        Concurrent means neither dominates the other.
        """
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        
        if clock_a.is_concurrent_with(clock_b):
            # There must exist at least one node where A > B
            a_gt_b_exists = any(
                clock_a.get_timestamp(n) > clock_b.get_timestamp(n)
                for n in set(dict_a.keys()) | set(dict_b.keys())
            )
            # And at least one node where B > A
            b_gt_a_exists = any(
                clock_b.get_timestamp(n) > clock_a.get_timestamp(n)
                for n in set(dict_a.keys()) | set(dict_b.keys())
            )
            
            assert a_gt_b_exists and b_gt_a_exists, \
                f"Concurrent clocks should be incomparable: {dict_a} || {dict_b}"


class TestDistributedSystemProperties:
    """
    Properties that must hold for the distributed system as a whole.
    """
    
    @given(st.lists(st.tuples(node_ids, st.integers(0, 10)), min_size=1, max_size=20))
    def test_local_events_are_totally_ordered(self, operations):
        """
        On a single node, all events should be totally ordered.
        
        For any sequence of operations on one node, the resulting
        vector clocks should form a chain (each happens-before the next).
        """
        node = Node("test_node")
        events = []
        
        for op_id, value in operations:
            event = node.write(f"key_{op_id}", value)
            events.append(event)
        
        # All events should be totally ordered
        for i in range(len(events) - 1):
            assert events[i].vector_clock.happens_before(events[i+1].vector_clock), \
                f"Events not ordered: event {i} should happen before event {i+1}"
    
    @given(st.lists(st.tuples(node_ids, node_ids), min_size=1, max_size=10))
    def test_causality_chain_property(self, message_chain):
        """
        If there's a connected chain of messages A -> B -> C -> D, then
        events at A causally precede events at D.
        
        Note: We build a connected chain where each receiver becomes the next sender.
        """
        cluster = Cluster()
        
        # Filter out self-loops and build a connected chain
        filtered = [(s, r) for s, r in message_chain if s != r]
        if not filtered:
            return
        
        # Build connected chain: receiver of step i becomes sender of step i+1
        connected_chain = [filtered[0]]
        last_receiver = filtered[0][1]
        for sender, receiver in filtered[1:]:
            if sender == last_receiver:
                connected_chain.append((sender, receiver))
                last_receiver = receiver
        
        # Need at least 2 hops for meaningful causality chain
        if len(connected_chain) < 2:
            return
        
        # Create nodes
        nodes = {}
        for sender, receiver in connected_chain:
            if sender not in nodes:
                nodes[sender] = cluster.add_node(sender)
            if receiver not in nodes:
                nodes[receiver] = cluster.add_node(receiver)
        
        # First node writes
        first_sender, first_receiver = connected_chain[0]
        first_node = nodes[first_sender]
        first_event = first_node.write("test_key", 0)
        
        # Follow the connected chain
        for sender_id, receiver_id in connected_chain:
            sender = nodes[sender_id]
            receiver = nodes[receiver_id]
            sender.send_message(receiver, "test_key")
        
        # Last receiver should have causally followed the first write
        last_receiver_node = nodes[connected_chain[-1][1]]
        last_event = last_receiver_node.write("final", 1)
        assert last_event.vector_clock.happens_after(first_event.vector_clock), \
            "Causality chain broken"
    
    @given(st.integers(min_value=2, max_value=5), st.integers(min_value=1, max_value=10))
    def test_partition_creates_concurrent_events(self, num_nodes, num_writes):
        """
        During a partition, writes on different partitions are concurrent.
        """
        cluster = Cluster()
        nodes = [cluster.add_node(f"N{i}") for i in range(num_nodes)]
        
        # Each node writes (simulating partition where no one can communicate)
        events = [node.write("x", i) for i, node in enumerate(nodes)]
        
        # All pairs of events from different nodes should be concurrent
        for i, event_i in enumerate(events):
            for j, event_j in enumerate(events):
                if i != j:
                    assert event_i.vector_clock.is_concurrent_with(event_j.vector_clock), \
                        f"Events from different nodes should be concurrent during partition"


class TestClockArithmetic:
    """
    Properties of clock arithmetic operations.
    """
    
    @given(vector_clock_dicts())
    def test_empty_clock_is_identity(self, dict_a):
        """
        For all clocks A: A.merge(empty) == A
        """
        clock_a = VectorClock.from_dict(dict_a)
        empty = VectorClock()
        
        merged = clock_a.merge(empty)
        assert merged.is_equal(clock_a)
    
    @given(vector_clock_dicts(), vector_clock_dicts(), st.data())
    def test_increment_then_merge_properties(self, dict_a, dict_b, data):
        """
        Test properties of increment and merge interaction.
        
        Note: (A.increment(n)).merge(B) and A.merge(B).increment(n) are NOT
        necessarily equal, because:
        - Path 1: A[n] becomes A[n]+1, then max with B[n]
        - Path 2: max(A[n], B[n]), then +1
        
        We verify the correct relationship instead.
        """
        assume(len(dict_a) > 0)
        
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        node = data.draw(st.sampled_from(list(dict_a.keys())))
        
        # Path 1: increment A, then merge with B
        path1 = clock_a.increment(node).merge(clock_b)
        
        # Path 2: merge A with B, then increment
        path2 = clock_a.merge(clock_b).increment(node)
        
        # Path 2's node value should always be >= Path 1's
        # Because: max(A,B)+1 >= max(A+1, B) when A is the max
        # But if B[n] > A[n]+1, then path1 might have higher value
        
        # What we know for sure:
        # - Both paths dominate the original A
        assert path1.happens_after(clock_a) or path1.is_equal(clock_a)
        assert path2.happens_after(clock_a) or path2.is_equal(clock_a)
        
        # - Both paths dominate B
        assert path1.happens_after(clock_b) or path1.is_equal(clock_b)
        assert path2.happens_after(clock_b) or path2.is_equal(clock_b)


class TestCausalityInvariants:
    """
    Critical invariants that must never be violated.
    """
    
    @given(vector_clock_dicts(), vector_clock_dicts())
    def test_no_cycles_in_causality(self, dict_a, dict_b):
        """
        Causality must be acyclic.
        
        It's impossible for A -> B and B -> A simultaneously.
        """
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        
        a_before_b = clock_a.happens_before(clock_b)
        b_before_a = clock_b.happens_before(clock_a)
        
        assert not (a_before_b and b_before_a), \
            f"Causality cycle detected: {dict_a} <-> {dict_b}"
    
    @given(vector_clock_dicts(), vector_clock_dicts(), vector_clock_dicts())
    def test_concurrent_is_not_transitive(self, dict_a, dict_b, dict_c):
        """
        Concurrency is NOT transitive.
        
        A || B and B || C does NOT imply A || C.
        This is a known property of vector clocks.
        """
        # This test just documents that concurrency is not transitive
        # We don't assert anything because it's valid for A || C or not
        clock_a = VectorClock.from_dict(dict_a)
        clock_b = VectorClock.from_dict(dict_b)
        clock_c = VectorClock.from_dict(dict_c)
        
        a_concurrent_b = clock_a.is_concurrent_with(clock_b)
        b_concurrent_c = clock_b.is_concurrent_with(clock_c)
        
        # Just verify the clocks are valid - no invariant to check here
        # This test serves as documentation
        if a_concurrent_b and b_concurrent_c:
            # A and C may or may not be concurrent - both are valid
            pass
