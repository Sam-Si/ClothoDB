"""
Exhaustive unit tests for Vector Clock implementation.

These tests verify the fundamental properties of vector clocks:
1. Basic operations (increment, merge)
2. Causality comparison (happens-before, concurrent, equal)
3. Edge cases and boundary conditions
4. Mathematical properties (transitivity, antisymmetry, etc.)
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from clotho.vector_clock import VectorClock, CausalityRelation


class TestVectorClockConstruction:
    """Tests for creating vector clocks."""
    
    def test_new_creates_clock_with_zero(self):
        """Creating a new clock should initialize the node counter to 0."""
        vc = VectorClock.new("node_a")
        assert vc.get_timestamp("node_a") == 0
        assert len(vc) == 1
    
    def test_from_dict_creates_clock(self):
        """Should be able to create from a dictionary."""
        vc = VectorClock.from_dict({"a": 1, "b": 2, "c": 3})
        assert vc.get_timestamp("a") == 1
        assert vc.get_timestamp("b") == 2
        assert vc.get_timestamp("c") == 3
    
    def test_from_dict_isolated_from_original(self):
        """Modifying original dict shouldn't affect the clock."""
        original = {"a": 1, "b": 2}
        vc = VectorClock.from_dict(original)
        original["a"] = 999
        assert vc.get_timestamp("a") == 1
    
    def test_empty_clock_creation(self):
        """Empty clock should be valid."""
        vc = VectorClock()
        assert len(vc) == 0
        assert vc.nodes() == set()
    
    def test_invalid_timestamp_negative(self):
        """Negative timestamps should raise ValueError."""
        with pytest.raises(ValueError):
            VectorClock(clock={"a": -1})
    
    def test_invalid_timestamp_non_int(self):
        """Non-integer timestamps should raise ValueError."""
        with pytest.raises(ValueError):
            VectorClock(clock={"a": "string"})


class TestVectorClockIncrement:
    """Tests for the increment operation."""
    
    def test_increment_creates_new_clock(self):
        """Increment should return a new clock (immutable)."""
        vc = VectorClock.new("node_a")
        vc2 = vc.increment("node_a")
        assert vc is not vc2
        assert vc.get_timestamp("node_a") == 0
        assert vc2.get_timestamp("node_a") == 1
    
    def test_increment_new_node(self):
        """Incrementing a new node adds it to the clock."""
        vc = VectorClock.new("node_a")
        vc2 = vc.increment("node_b")
        assert vc2.get_timestamp("node_a") == 0
        assert vc2.get_timestamp("node_b") == 1
    
    def test_multiple_increments(self):
        """Multiple increments should accumulate."""
        vc = VectorClock.new("node_a")
        for i in range(10):
            vc = vc.increment("node_a")
        assert vc.get_timestamp("node_a") == 10
    
    def test_increment_preserves_other_nodes(self):
        """Incrementing one node shouldn't affect others."""
        vc = VectorClock.from_dict({"a": 5, "b": 3, "c": 7})
        vc2 = vc.increment("b")
        assert vc2.get_timestamp("a") == 5
        assert vc2.get_timestamp("b") == 4
        assert vc2.get_timestamp("c") == 7


class TestVectorClockMerge:
    """Tests for the merge operation (component-wise max)."""
    
    def test_merge_creates_new_clock(self):
        """Merge should return a new clock."""
        vc1 = VectorClock.from_dict({"a": 1})
        vc2 = VectorClock.from_dict({"a": 2})
        vc3 = vc1.merge(vc2)
        assert vc3 is not vc1
        assert vc3 is not vc2
    
    def test_merge_disjoint_clocks(self):
        """Merging disjoint clocks combines them."""
        vc1 = VectorClock.from_dict({"a": 5})
        vc2 = VectorClock.from_dict({"b": 3})
        vc3 = vc1.merge(vc2)
        assert vc3.get_timestamp("a") == 5
        assert vc3.get_timestamp("b") == 3
    
    def test_merge_overlapping_clocks(self):
        """Merge takes max for overlapping nodes."""
        vc1 = VectorClock.from_dict({"a": 5, "b": 2})
        vc2 = VectorClock.from_dict({"a": 3, "b": 7})
        vc3 = vc1.merge(vc2)
        assert vc3.get_timestamp("a") == 5  # max(5, 3)
        assert vc3.get_timestamp("b") == 7  # max(2, 7)
    
    def test_merge_three_clocks(self):
        """Merge should be associative."""
        vc1 = VectorClock.from_dict({"a": 1, "b": 2})
        vc2 = VectorClock.from_dict({"b": 3, "c": 4})
        vc3 = VectorClock.from_dict({"c": 2, "d": 5})
        
        # (vc1 merge vc2) merge vc3
        merge1 = vc1.merge(vc2).merge(vc3)
        # vc1 merge (vc2 merge vc3)
        merge2 = vc1.merge(vc2.merge(vc3))
        
        assert merge1.clock == merge2.clock
        assert merge1.get_timestamp("a") == 1
        assert merge1.get_timestamp("b") == 3
        assert merge1.get_timestamp("c") == 4
        assert merge1.get_timestamp("d") == 5
    
    def test_merge_with_empty_clock(self):
        """Merging with empty clock should return the other."""
        vc1 = VectorClock.from_dict({"a": 5, "b": 3})
        vc_empty = VectorClock()
        
        assert vc1.merge(vc_empty).clock == vc1.clock
        assert vc_empty.merge(vc1).clock == vc1.clock
    
    def test_merge_idempotent(self):
        """Merging a clock with itself should return the same clock."""
        vc = VectorClock.from_dict({"a": 5, "b": 3, "c": 7})
        merged = vc.merge(vc)
        assert merged.clock == vc.clock
    
    def test_merge_commutative(self):
        """Merge should be commutative: a.merge(b) == b.merge(a)."""
        vc1 = VectorClock.from_dict({"a": 5, "b": 2, "c": 8})
        vc2 = VectorClock.from_dict({"b": 7, "c": 3, "d": 4})
        
        merge1 = vc1.merge(vc2)
        merge2 = vc2.merge(vc1)
        
        assert merge1.clock == merge2.clock


class TestVectorClockComparisonEqual:
    """Tests for EQUAL causality relation."""
    
    def test_equal_same_clock(self):
        """A clock should be equal to itself."""
        vc = VectorClock.from_dict({"a": 1, "b": 2})
        assert vc.compare(vc) == CausalityRelation.EQUAL
        assert vc.is_equal(vc)
    
    def test_equal_identical_clocks(self):
        """Two clocks with same values should be equal."""
        vc1 = VectorClock.from_dict({"a": 1, "b": 2})
        vc2 = VectorClock.from_dict({"a": 1, "b": 2})
        assert vc1.compare(vc2) == CausalityRelation.EQUAL
        assert vc1.is_equal(vc2)
    
    def test_equal_different_order(self):
        """Dict order shouldn't matter for equality."""
        vc1 = VectorClock.from_dict({"a": 1, "b": 2, "c": 3})
        vc2 = VectorClock.from_dict({"c": 3, "a": 1, "b": 2})
        assert vc1.compare(vc2) == CausalityRelation.EQUAL
    
    def test_not_equal_different_values(self):
        """Clocks with different values are not equal."""
        vc1 = VectorClock.from_dict({"a": 1, "b": 2})
        vc2 = VectorClock.from_dict({"a": 1, "b": 3})
        assert vc1.compare(vc2) != CausalityRelation.EQUAL
        assert not vc1.is_equal(vc2)


class TestVectorClockComparisonHappensBefore:
    """Tests for HAPPENS_BEFORE causality relation."""
    
    def test_simple_happens_before(self):
        """If clock A has all values <= B and at least one <, A happens before B."""
        vc_a = VectorClock.from_dict({"a": 1, "b": 2})
        vc_b = VectorClock.from_dict({"a": 2, "b": 3})
        assert vc_a.compare(vc_b) == CausalityRelation.HAPPENS_BEFORE
        assert vc_a.happens_before(vc_b)
        assert vc_b.happens_after(vc_a)
    
    def test_happens_before_with_extra_nodes(self):
        """A happens before B even if B has extra nodes."""
        vc_a = VectorClock.from_dict({"a": 1})
        vc_b = VectorClock.from_dict({"a": 1, "b": 1})
        assert vc_a.compare(vc_b) == CausalityRelation.HAPPENS_BEFORE
    
    def test_happens_before_strict_less(self):
        """For happens_before, at least one component must be strictly less."""
        vc_a = VectorClock.from_dict({"a": 1, "b": 2})
        vc_b = VectorClock.from_dict({"a": 1, "b": 3})  # a same, b greater
        assert vc_a.compare(vc_b) == CausalityRelation.HAPPENS_BEFORE
    
    def test_not_happens_before_when_concurrent(self):
        """Concurrent clocks don't have happens_before relationship."""
        vc_a = VectorClock.from_dict({"a": 2, "b": 1})
        vc_b = VectorClock.from_dict({"a": 1, "b": 2})
        assert not vc_a.happens_before(vc_b)
        assert not vc_b.happens_before(vc_a)


class TestVectorClockComparisonConcurrent:
    """Tests for CONCURRENT causality relation."""
    
    def test_simple_concurrent(self):
        """
        Classic concurrent case:
        - A has a=2, b=1 (A did 2 events, saw 1 from B)
        - B has a=1, b=2 (B did 2 events, saw 1 from A)
        Neither dominates the other -> concurrent
        """
        vc_a = VectorClock.from_dict({"a": 2, "b": 1})
        vc_b = VectorClock.from_dict({"a": 1, "b": 2})
        assert vc_a.compare(vc_b) == CausalityRelation.CONCURRENT
        assert vc_a.is_concurrent_with(vc_b)
    
    def test_concurrent_with_disjoint_nodes(self):
        """Clocks with completely different nodes are concurrent."""
        vc_a = VectorClock.from_dict({"a": 5, "b": 3})
        vc_c = VectorClock.from_dict({"c": 2, "d": 4})
        assert vc_a.compare(vc_c) == CausalityRelation.CONCURRENT
    
    def test_concurrent_partial_overlap(self):
        """Concurrent with partial overlap."""
        vc_a = VectorClock.from_dict({"a": 3, "b": 1, "c": 2})
        vc_b = VectorClock.from_dict({"a": 2, "b": 3, "c": 1})
        # a: 3>2, b: 1<3, c: 2>1 -> neither dominates
        assert vc_a.compare(vc_b) == CausalityRelation.CONCURRENT
    
    def test_concurrent_symmetric(self):
        """Concurrency is symmetric."""
        vc_a = VectorClock.from_dict({"a": 2, "b": 1})
        vc_b = VectorClock.from_dict({"a": 1, "b": 2})
        assert vc_a.is_concurrent_with(vc_b)
        assert vc_b.is_concurrent_with(vc_a)


class TestVectorClockComparisonEdgeCases:
    """Edge cases for causality comparison."""
    
    def test_empty_vs_empty(self):
        """Two empty clocks are equal."""
        vc1 = VectorClock()
        vc2 = VectorClock()
        assert vc1.compare(vc2) == CausalityRelation.EQUAL
    
    def test_empty_vs_nonempty(self):
        """Empty clock happens before non-empty clock."""
        vc_empty = VectorClock()
        vc_full = VectorClock.from_dict({"a": 1})
        assert vc_empty.compare(vc_full) == CausalityRelation.HAPPENS_BEFORE
        assert vc_full.compare(vc_empty) == CausalityRelation.HAPPENS_AFTER
    
    def test_zero_vs_positive(self):
        """Clock with zeros happens before clock with positive values."""
        vc1 = VectorClock.from_dict({"a": 0, "b": 0})
        vc2 = VectorClock.from_dict({"a": 0, "b": 1})
        assert vc1.compare(vc2) == CausalityRelation.HAPPENS_BEFORE
    
    def test_missing_node_treated_as_zero(self):
        """Missing nodes are treated as having timestamp 0."""
        vc1 = VectorClock.from_dict({"a": 1})  # b is implicitly 0
        vc2 = VectorClock.from_dict({"a": 1, "b": 0})  # b is explicit 0
        assert vc1.compare(vc2) == CausalityRelation.EQUAL


class TestVectorClockMathematicalProperties:
    """
    Mathematical properties that vector clocks must satisfy.
    
    These are invariants that must hold for the implementation to be correct.
    """
    
    def test_comparison_antisymmetric(self):
        """
        If A happens before B, then B cannot happen before A.
        """
        vc1 = VectorClock.from_dict({"a": 1, "b": 2})
        vc2 = VectorClock.from_dict({"a": 2, "b": 3})
        
        assert vc1.happens_before(vc2)
        assert not vc2.happens_before(vc1)
    
    def test_comparison_transitive(self):
        """
        Transitivity: if A -> B and B -> C, then A -> C.
        """
        vc_a = VectorClock.from_dict({"x": 1})
        vc_b = VectorClock.from_dict({"x": 2})
        vc_c = VectorClock.from_dict({"x": 3})
        
        assert vc_a.happens_before(vc_b)
        assert vc_b.happens_before(vc_c)
        assert vc_a.happens_before(vc_c)
    
    def test_merge_preserves_order(self):
        """
        If A happens before B, then:
        - A.merge(B) == B (B dominates)
        - A.merge(B) happens after A
        """
        vc_a = VectorClock.from_dict({"x": 1, "y": 1})
        vc_b = VectorClock.from_dict({"x": 2, "y": 2})
        
        merged = vc_a.merge(vc_b)
        
        assert merged.clock == vc_b.clock
        assert merged.happens_after(vc_a)
    
    def test_concurrent_incomparable(self):
        """
        Concurrent events are incomparable: neither happens before the other.
        """
        vc_a = VectorClock.from_dict({"a": 2, "b": 1})
        vc_b = VectorClock.from_dict({"a": 1, "b": 2})
        
        assert vc_a.is_concurrent_with(vc_b)
        assert not vc_a.happens_before(vc_b)
        assert not vc_b.happens_before(vc_a)
        assert not vc_a.happens_after(vc_b)  # happens_after is inverse
        assert not vc_b.happens_after(vc_a)
    
    def test_equality_reflexive(self):
        """Equality is reflexive: A == A."""
        vc = VectorClock.from_dict({"a": 1, "b": 2, "c": 3})
        assert vc.is_equal(vc)
    
    def test_equality_symmetric(self):
        """Equality is symmetric: if A == B then B == A."""
        vc1 = VectorClock.from_dict({"a": 1, "b": 2})
        vc2 = VectorClock.from_dict({"a": 1, "b": 2})
        assert vc1.is_equal(vc2)
        assert vc2.is_equal(vc1)
    
    def test_equality_transitive(self):
        """Equality is transitive: if A == B and B == C, then A == C."""
        vc1 = VectorClock.from_dict({"a": 1})
        vc2 = VectorClock.from_dict({"a": 1})
        vc3 = VectorClock.from_dict({"a": 1})
        assert vc1.is_equal(vc2)
        assert vc2.is_equal(vc3)
        assert vc1.is_equal(vc3)


class TestVectorClockSerialization:
    """Tests for serialization/deserialization."""
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        vc = VectorClock.from_dict({"a": 1, "b": 2})
        d = vc.to_dict()
        assert d == {"a": 1, "b": 2}
    
    def test_to_dict_isolated(self):
        """Modifying returned dict shouldn't affect clock."""
        vc = VectorClock.from_dict({"a": 1, "b": 2})
        d = vc.to_dict()
        d["a"] = 999
        assert vc.get_timestamp("a") == 1
    
    def test_round_trip(self):
        """to_dict -> from_dict should preserve clock."""
        original = VectorClock.from_dict({"a": 1, "b": 2, "c": 3})
        d = original.to_dict()
        restored = VectorClock.from_dict(d)
        assert original.is_equal(restored)


class TestVectorClockImmutability:
    """Tests to verify vector clock immutability."""
    
    def test_increment_doesnt_modify_original(self):
        """Increment should not modify the original clock."""
        vc = VectorClock.from_dict({"a": 1})
        vc2 = vc.increment("a")
        assert vc.get_timestamp("a") == 1
        assert vc2.get_timestamp("a") == 2
    
    def test_merge_doesnt_modify_originals(self):
        """Merge should not modify either original clock."""
        vc1 = VectorClock.from_dict({"a": 1})
        vc2 = VectorClock.from_dict({"b": 2})
        vc3 = vc1.merge(vc2)
        assert vc1.get_timestamp("a") == 1
        assert "b" not in vc1.clock
        assert vc2.get_timestamp("b") == 2
        assert "a" not in vc2.clock
    
    def test_frozen_dataclass_prevents_reassignment(self):
        """Frozen dataclass prevents reassignment of clock attribute."""
        vc = VectorClock.from_dict({"a": 1})
        # Cannot reassign the clock attribute
        with pytest.raises(Exception):
            vc.clock = {"a": 999}
    
    def test_api_immutability(self):
        """
        The public API is immutable - operations return new clocks.
        
        This is the key guarantee: increment() and merge() don't modify
        the original clock, they return new ones.
        """
        vc = VectorClock.from_dict({"a": 1})
        
        # Increment returns new clock
        vc2 = vc.increment("a")
        assert vc.get_timestamp("a") == 1  # Original unchanged
        assert vc2.get_timestamp("a") == 2  # New has incremented value
        
        # Merge returns new clock
        other = VectorClock.from_dict({"b": 5})
        vc3 = vc.merge(other)
        assert "b" not in vc.clock  # Original unchanged
        assert vc3.get_timestamp("b") == 5  # New has merged value


class TestVectorClockUtilityMethods:
    """Tests for utility methods."""
    
    def test_nodes_returns_all_node_ids(self):
        """nodes() should return all node IDs."""
        vc = VectorClock.from_dict({"a": 1, "b": 2, "c": 3})
        assert vc.nodes() == {"a", "b", "c"}
    
    def test_contains_checks_node_existence(self):
        """__contains__ should check if node exists."""
        vc = VectorClock.from_dict({"a": 1, "b": 2})
        assert "a" in vc
        assert "c" not in vc
    
    def test_len_returns_node_count(self):
        """__len__ should return number of nodes."""
        vc = VectorClock.from_dict({"a": 1, "b": 2, "c": 3})
        assert len(vc) == 3
    
    def test_get_timestamp_missing_returns_zero(self):
        """get_timestamp should return 0 for missing nodes."""
        vc = VectorClock.from_dict({"a": 5})
        assert vc.get_timestamp("z") == 0
    
    def test_copy_creates_equal_clock(self):
        """copy() should create an equal but separate clock."""
        vc = VectorClock.from_dict({"a": 1, "b": 2})
        vc_copy = vc.copy()
        assert vc.is_equal(vc_copy)
        assert vc is not vc_copy
    
    def test_hash_consistent(self):
        """Equal clocks should have equal hashes."""
        vc1 = VectorClock.from_dict({"a": 1, "b": 2})
        vc2 = VectorClock.from_dict({"a": 1, "b": 2})
        assert hash(vc1) == hash(vc2)
    
    def test_repr_format(self):
        """repr should be readable and consistent."""
        vc = VectorClock.from_dict({"b": 2, "a": 1})
        repr_str = repr(vc)
        assert "VC(" in repr_str
        assert "a:1" in repr_str
        assert "b:2" in repr_str
