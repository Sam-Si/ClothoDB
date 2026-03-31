"""
Distributed Node implementation with Vector Clock tracking.

This module simulates nodes in a leaderless distributed database,
where each node maintains its own vector clock to track causality.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime

from .vector_clock import VectorClock, CausalityRelation


@dataclass(frozen=True)
class Event:
    """
    An event in the distributed system with an associated vector clock.
    
    Events are immutable and uniquely identified. The vector clock
tells us the causal history of this event.
    """
    event_id: str
    node_id: str
    vector_clock: VectorClock
    operation: str  # e.g., "WRITE", "READ", "DELETE"
    key: str        # the database key being operated on
    value: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def create(
        cls,
        node_id: str,
        vector_clock: VectorClock,
        operation: str,
        key: str,
        value: Optional[Any] = None
    ) -> Event:
        """Factory method to create a new event with a unique ID."""
        return cls(
            event_id=str(uuid.uuid4()),
            node_id=node_id,
            vector_clock=vector_clock,
            operation=operation,
            key=key,
            value=value
        )
    
    def __repr__(self) -> str:
        return f"Event({self.event_id[:8]}, {self.node_id}, {self.operation}, {self.key}, {self.vector_clock})"


class Node:
    """
    A node in the distributed database cluster.
    
    Each node maintains:
    1. Its own vector clock tracking all known events across the cluster
    2. A local event history
    3. A key-value store (simplified)
    
    Key Design Decisions:
    ---------------------
    1. Vector clocks grow with the number of nodes (O(n) space per event)
    2. When nodes communicate, they exchange and merge vector clocks
    3. Causality is determined by comparing vector clocks
    
    The node_id is the unique identifier for this node in the cluster.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        # Start with a fresh vector clock for this node
        self.vector_clock = VectorClock.new(node_id)
        # Local event history
        self.events: List[Event] = []
        # Simple key-value store: key -> (value, vector_clock)
        self.data: Dict[str, tuple] = {}
        # Known nodes in the cluster (grows as we communicate)
        self.known_nodes: Set[str] = {node_id}
        # Event handlers for simulation/testing
        self._event_handlers: List[Callable[[Event], None]] = []
    
    def _emit_event(self, event: Event) -> None:
        """Emit an event to all registered handlers."""
        for handler in self._event_handlers:
            handler(event)
    
    def add_event_handler(self, handler: Callable[[Event], None]) -> None:
        """Add a handler to be called on every local event."""
        self._event_handlers.append(handler)
    
    def perform_operation(
        self,
        operation: str,
        key: str,
        value: Optional[Any] = None
    ) -> Event:
        """
        Perform a local operation and create an event.
        
        Steps:
        1. Increment own vector clock (new event)
        2. Create event with new clock
        3. Store event in local history
        4. Update local data store
        5. Emit event
        """
        # Increment our own counter
        self.vector_clock = self.vector_clock.increment(self.node_id)
        
        # Create the event
        event = Event.create(
            node_id=self.node_id,
            vector_clock=self.vector_clock,
            operation=operation,
            key=key,
            value=value
        )
        
        # Store event
        self.events.append(event)
        
        # Update local data
        if operation in ("WRITE", "DELETE"):
            self.data[key] = (value, self.vector_clock)
        
        # Emit
        self._emit_event(event)
        
        return event
    
    def write(self, key: str, value: Any) -> Event:
        """Write a key-value pair."""
        return self.perform_operation("WRITE", key, value)
    
    def delete(self, key: str) -> Event:
        """Delete a key."""
        return self.perform_operation("DELETE", key, None)
    
    def read(self, key: str) -> Optional[tuple]:
        """Read a key's value and vector clock."""
        return self.data.get(key)
    
    def receive_message(self, sender_node_id: str, sender_clock: VectorClock) -> None:
        """
        Receive a message (and its vector clock) from another node.
        
        This is the core of causality tracking in distributed systems.
        When we receive a message:
        1. We learn about the sender's events (merge clocks)
        2. We record that we received a message (increment own clock)
        3. We now know about all nodes the sender knows about
        
        The merge operation ensures causality is preserved:
        - If sender's event A causally precedes our event B,
          after merge, the causal relationship is maintained
        """
        # Learn about all nodes the sender knows
        self.known_nodes.update(sender_clock.nodes())
        self.known_nodes.add(sender_node_id)
        
        # Merge the sender's clock with ours
        # This captures: "we now know everything the sender knew"
        self.vector_clock = self.vector_clock.merge(sender_clock)
        
        # Increment our own clock to record this receive event
        self.vector_clock = self.vector_clock.increment(self.node_id)
    
    def send_message(self, recipient: Node, key: str) -> VectorClock:
        """
        Send a message to another node.
        
        Returns the vector clock sent (for testing/verification).
        """
        # Get current value for the key
        value_info = self.data.get(key)
        if value_info:
            _, value_clock = value_info
            # Include the value's clock in what we send
            send_clock = self.vector_clock.merge(value_clock)
        else:
            send_clock = self.vector_clock
        
        # Recipient receives our clock
        recipient.receive_message(self.node_id, send_clock)
        
        return send_clock
    
    def compare_with_node(self, other: Node) -> CausalityRelation:
        """Compare this node's current vector clock with another node's."""
        return self.vector_clock.compare(other.vector_clock)
    
    def is_causally_ready(self, event: Event) -> bool:
        """
        Check if we have received all causally preceding events
        for the given event.
        
        An event E is causally ready at node N if:
        N.vector_clock >= E.vector_clock (excluding E's own increment)
        
        This is crucial for ensuring causal consistency.
        """
        # The event's clock should not dominate our clock
        # (we should know about everything that happened before this event)
        relation = self.vector_clock.compare(event.vector_clock)
        
        # We're ready if we happen after or are equal to the event
        # (meaning we've seen all events that causally precede it)
        return relation in (CausalityRelation.HAPPENS_AFTER, CausalityRelation.EQUAL)
    
    def get_causal_history(self) -> List[Event]:
        """Get all events in causal order (based on vector clock comparison)."""
        # Sort events by vector clock (causal order)
        # Note: concurrent events may have arbitrary order
        return sorted(
            self.events,
            key=lambda e: (sum(e.vector_clock.clock.values()), e.timestamp)
        )
    
    def get_concurrent_events(self) -> List[List[Event]]:
        """
        Find groups of concurrent events in our history.
        
        Returns lists of events where all events in each list
        are concurrent with each other.
        """
        concurrent_groups = []
        
        for event in self.events:
            added = False
            for group in concurrent_groups:
                # Check if concurrent with all events in group
                if all(
                    event.vector_clock.is_concurrent_with(e.vector_clock)
                    for e in group
                ):
                    group.append(event)
                    added = True
                    break
            if not added:
                concurrent_groups.append([event])
        
        return concurrent_groups
    
    def __repr__(self) -> str:
        return f"Node({self.node_id}, clock={self.vector_clock})"


class Cluster:
    """
    A cluster of nodes for simulation/testing.
    
    This class helps coordinate multiple nodes and provides
    utilities for testing distributed scenarios.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self._all_events: List[Event] = []
    
    def add_node(self, node_id: str) -> Node:
        """Add a new node to the cluster."""
        node = Node(node_id)
        self.nodes[node_id] = node
        node.add_event_handler(self._all_events.append)
        return node
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def partition(self, partition1: Set[str], partition2: Set[str]) -> None:
        """
        Simulate a network partition between two sets of nodes.
        
        During a partition, nodes in different partitions cannot communicate.
        This leads to divergent vector clocks and concurrent events.
        """
        # In a real implementation, this would block communication
        # For simulation, we just track the partition
        self._partition1 = partition1
        self._partition2 = partition2
    
    def heal_partition(self) -> None:
        """Heal a network partition, allowing all nodes to communicate."""
        self._partition1 = None
        self._partition2 = None
    
    def get_global_causal_order(self) -> List[Event]:
        """
        Get a global causal ordering of all events across all nodes.
        
        This uses vector clocks to establish a partial order.
        Concurrent events are grouped together.
        """
        all_events = []
        for node in self.nodes.values():
            all_events.extend(node.events)
        
        # Topological sort based on vector clock comparison
        sorted_events = []
        remaining = set(range(len(all_events)))
        
        while remaining:
            # Find events with no unprocessed predecessors
            ready = []
            for i in remaining:
                event = all_events[i]
                has_predecessor = False
                for j in remaining:
                    if i != j:
                        other = all_events[j]
                        if other.vector_clock.happens_before(event.vector_clock):
                            has_predecessor = True
                            break
                if not has_predecessor:
                    ready.append(i)
            
            if not ready:
                # Cycle detected (shouldn't happen with vector clocks)
                break
            
            for i in ready:
                sorted_events.append(all_events[i])
                remaining.remove(i)
        
        return sorted_events
    
    def find_concurrent_conflicts(self) -> List[Tuple[Event, Event]]:
        """
        Find all pairs of concurrent events that might be conflicts.
        
        Returns pairs of events where:
        1. They are concurrent (no causal relationship)
        2. They operate on the same key
        3. Both are WRITE operations
        """
        conflicts = []
        all_events = self.get_global_causal_order()
        
        for i, event1 in enumerate(all_events):
            for event2 in all_events[i+1:]:
                if (
                    event1.key == event2.key and
                    event1.operation == "WRITE" and
                    event2.operation == "WRITE" and
                    event1.vector_clock.is_concurrent_with(event2.vector_clock)
                ):
                    conflicts.append((event1, event2))
        
        return conflicts
    
    def __repr__(self) -> str:
        return f"Cluster(nodes={list(self.nodes.keys())})"
