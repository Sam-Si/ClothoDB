"""
Vector Clock implementation for distributed causality tracking.

Vector clocks solve the fundamental problem: given two events A and B
on different nodes, determine if A happened before B, B happened before A,
or if they are concurrent (no causal relationship).

Key insight: Each node maintains a vector of counters, one per node in the system.
When node i performs an event, it increments its own counter VC[i].
When nodes communicate, they merge clocks by taking component-wise maximum.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Set, Tuple


class CausalityRelation(Enum):
    """
    The possible causal relationships between two events.
    
    HAPPENS_BEFORE: Event A causally precedes Event B
    HAPPENS_AFTER: Event B causally precedes Event A  
    CONCURRENT: No causal relationship - events happened independently
    EQUAL: Same event (identical vector clocks)
    """
    HAPPENS_BEFORE = auto()  # A -> B (A causally precedes B)
    HAPPENS_AFTER = auto()   # B -> A (B causally precedes A)
    CONCURRENT = auto()      # A || B (concurrent, no causal relationship)
    EQUAL = auto()           # A == B (same vector clock)


@dataclass(frozen=True)
class VectorClock:
    """
    Immutable vector clock for tracking causality in distributed systems.
    
    A vector clock VC is a map: node_id -> logical_counter
    
    Causality Rules:
    ----------------
    VC1 <= VC2 iff for all nodes n: VC1[n] <= VC2[n]
    
    - VC1 HAPPENS_BEFORE VC2 if VC1 <= VC2 AND VC1 != VC2
    - VC1 CONCURRENT VC2 if NOT(VC1 <= VC2) AND NOT(VC2 <= VC1)
    - VC1 EQUAL VC2 if VC1 == VC2 (component-wise equality)
    
    Example scenario with 3 nodes (A, B, C):
    ----------------------------------------
    Time →
    
    Node A: [1,0,0] → [2,0,0] → [3,2,1] (receives from B and C)
                  ↘
    Node B: [0,1,0] → [1,2,0] → [1,3,0]
                  ↗
    Node C: [0,0,1] ───────────→ [0,0,2]
    
    At [3,2,1] on A, we can see:
    - A has performed 3 events
    - B has performed 2 events (A has received 2 updates from B)
    - C has performed 1 event (A has received 1 update from C)
    """
    
    # Map of node_id -> logical timestamp
    # Using immutable dict pattern via frozen dataclass
    clock: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure all values are non-negative integers
        for node_id, timestamp in self.clock.items():
            if not isinstance(timestamp, int) or timestamp < 0:
                raise ValueError(f"Invalid timestamp for {node_id}: {timestamp}")
    
    @classmethod
    def new(cls, node_id: str) -> VectorClock:
        """Create a new vector clock for a single node starting at 0."""
        return cls(clock={node_id: 0})
    
    @classmethod
    def from_dict(cls, clock_dict: Dict[str, int]) -> VectorClock:
        """Create a vector clock from a dictionary."""
        return cls(clock=copy.deepcopy(clock_dict))
    
    def increment(self, node_id: str) -> VectorClock:
        """
        Increment the counter for a specific node.
        Called when 'node_id' performs a new local event.
        
        Returns a NEW vector clock (immutable operation).
        """
        new_clock = copy.deepcopy(self.clock)
        new_clock[node_id] = new_clock.get(node_id, 0) + 1
        return VectorClock(clock=new_clock)
    
    def merge(self, other: VectorClock) -> VectorClock:
        """
        Merge two vector clocks by taking component-wise maximum.
        Called when two nodes communicate (send/receive message).
        
        This is the CRITICAL operation for causality tracking.
        When node A receives a message from node B:
        1. A increments its own counter (new event: receiving)
        2. A merges its clock with B's clock (learns about B's history)
        
        Returns a NEW vector clock (immutable operation).
        """
        new_clock = copy.deepcopy(self.clock)
        
        for node_id, timestamp in other.clock.items():
            new_clock[node_id] = max(new_clock.get(node_id, 0), timestamp)
        
        return VectorClock(clock=new_clock)
    
    def compare(self, other: VectorClock) -> CausalityRelation:
        """
        Determine the causal relationship between this clock and another.
        
        This is the core algorithm for establishing happens-before relationships.
        
        Algorithm:
        ----------
        1. Check if all components of self <= other
        2. Check if all components of other <= self
        3. If both true: clocks are equal
        4. If only (1) true: self happens before other
        5. If only (2) true: other happens before self
        6. If neither: concurrent (no causal relationship)
        """
        # Get union of all node IDs
        all_nodes = set(self.clock.keys()) | set(other.clock.keys())
        
        # Track dominance relationships
        self_dominates = True   # self >= other (all components)
        other_dominates = True  # other >= self (all components)
        
        for node_id in all_nodes:
            self_ts = self.clock.get(node_id, 0)
            other_ts = other.clock.get(node_id, 0)
            
            if self_ts > other_ts:
                other_dominates = False
            elif other_ts > self_ts:
                self_dominates = False
            # If equal, both remain True
        
        # Determine relationship
        if self_dominates and other_dominates:
            return CausalityRelation.EQUAL
        elif self_dominates:
            # self >= other AND self != other (at least one component is strictly greater)
            # Need to check if they're actually different
            if self.clock == other.clock:
                return CausalityRelation.EQUAL
            return CausalityRelation.HAPPENS_AFTER
        elif other_dominates:
            return CausalityRelation.HAPPENS_BEFORE
        else:
            return CausalityRelation.CONCURRENT
    
    def happens_before(self, other: VectorClock) -> bool:
        """Check if this clock causally precedes the other."""
        return self.compare(other) == CausalityRelation.HAPPENS_BEFORE
    
    def happens_after(self, other: VectorClock) -> bool:
        """Check if this clock causally succeeds the other."""
        return self.compare(other) == CausalityRelation.HAPPENS_AFTER
    
    def is_concurrent_with(self, other: VectorClock) -> bool:
        """Check if this clock and other are concurrent (no causal relationship)."""
        return self.compare(other) == CausalityRelation.CONCURRENT
    
    def is_equal(self, other: VectorClock) -> bool:
        """Check if two clocks are equal."""
        return self.compare(other) == CausalityRelation.EQUAL
    
    def get_timestamp(self, node_id: str) -> int:
        """Get the timestamp for a specific node."""
        return self.clock.get(node_id, 0)
    
    def nodes(self) -> Set[str]:
        """Return all node IDs in this clock."""
        return set(self.clock.keys())
    
    def copy(self) -> VectorClock:
        """Return a deep copy of this clock."""
        return VectorClock(clock=copy.deepcopy(self.clock))
    
    def __len__(self) -> int:
        """Return the number of nodes in this clock."""
        return len(self.clock)
    
    def __contains__(self, node_id: str) -> bool:
        """Check if a node is in this clock."""
        return node_id in self.clock
    
    def __repr__(self) -> str:
        """String representation sorted by node_id for consistency."""
        items = sorted(self.clock.items())
        inner = ", ".join(f"{k}:{v}" for k, v in items)
        return f"VC({inner})"
    
    def __hash__(self) -> int:
        """Hash based on frozen clock contents."""
        return hash(frozenset(self.clock.items()))
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary representation."""
        return copy.deepcopy(self.clock)
