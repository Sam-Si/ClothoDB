"""
Visualization and debugging utilities for vector clocks.

These utilities help visualize causality relationships and debug
distributed system scenarios.
"""

from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

from .vector_clock import VectorClock, CausalityRelation
from .node import Node, Event, Cluster


def format_vector_clock(vc: VectorClock, node_order: Optional[List[str]] = None) -> str:
    """
    Format a vector clock as a pretty string.
    
    Example: "[A:3, B:1, C:2]"
    """
    if node_order:
        items = [(n, vc.get_timestamp(n)) for n in node_order if n in vc or vc.get_timestamp(n) > 0]
    else:
        items = sorted(vc.clock.items())
    
    inner = ", ".join(f"{k}:{v}" for k, v in items)
    return f"[{inner}]"


def format_causality_relation(relation: CausalityRelation) -> str:
    """Format a causality relation as a symbol."""
    symbols = {
        CausalityRelation.HAPPENS_BEFORE: "→",
        CausalityRelation.HAPPENS_AFTER: "←",
        CausalityRelation.CONCURRENT: "||",
        CausalityRelation.EQUAL: "==",
    }
    return symbols.get(relation, "?")


def create_causality_matrix(events: List[Event]) -> List[List[str]]:
    """
    Create a matrix showing causality relationships between events.
    
    Returns a 2D list where cell[i][j] shows the relationship
    between event i and event j.
    """
    n = len(events)
    matrix = [["" for _ in range(n + 1)] for _ in range(n + 1)]
    
    # Header row
    matrix[0][0] = ""
    for i, event in enumerate(events):
        matrix[0][i + 1] = f"E{i+1}"
        matrix[i + 1][0] = f"E{i+1}"
    
    # Fill in relationships
    for i, event_i in enumerate(events):
        for j, event_j in enumerate(events):
            relation = event_i.vector_clock.compare(event_j.vector_clock)
            matrix[i + 1][j + 1] = format_causality_relation(relation)
    
    return matrix


def print_causality_matrix(events: List[Event]) -> None:
    """Print a formatted causality matrix."""
    matrix = create_causality_matrix(events)
    
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in matrix) for i in range(len(matrix[0]))]
    
    for row in matrix:
        formatted = " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
        print(formatted)
        if row == matrix[0]:  # Print separator after header
            print("-" * len(formatted))


def visualize_event_history(node: Node, show_concurrent: bool = True) -> str:
    """
    Create a text visualization of a node's event history.
    
    Returns a formatted string showing events in causal order.
    """
    lines = [f"Event history for node {node.node_id}:", "=" * 50]
    
    events = node.get_causal_history()
    
    for i, event in enumerate(events):
        clock_str = format_vector_clock(event.vector_clock)
        lines.append(f"{i+1}. {event.operation} {event.key}={event.value}")
        lines.append(f"   Clock: {clock_str}")
        lines.append(f"   ID: {event.event_id[:8]}")
        
        if show_concurrent and i > 0:
            # Find concurrent events
            concurrent = []
            for j, prev_event in enumerate(events[:i]):
                if event.vector_clock.is_concurrent_with(prev_event.vector_clock):
                    concurrent.append(f"E{j+1}")
            if concurrent:
                lines.append(f"   Concurrent with: {', '.join(concurrent)}")
        
        lines.append("")
    
    return "\n".join(lines)


def visualize_cluster_state(cluster: Cluster) -> str:
    """
    Create a text visualization of the entire cluster state.
    """
    lines = ["Cluster State", "=" * 50]
    
    # Show each node's current clock
    lines.append("\nCurrent Vector Clocks:")
    lines.append("-" * 30)
    
    all_nodes = sorted(cluster.nodes.keys())
    for node_id in all_nodes:
        node = cluster.nodes[node_id]
        clock_str = format_vector_clock(node.vector_clock, all_nodes)
        lines.append(f"{node_id}: {clock_str}")
    
    # Show conflicts
    lines.append("\nConcurrent Conflicts:")
    lines.append("-" * 30)
    conflicts = cluster.find_concurrent_conflicts()
    if conflicts:
        for e1, e2 in conflicts:
            lines.append(f"  Conflict on '{e1.key}': {e1.node_id} vs {e2.node_id}")
    else:
        lines.append("  No conflicts detected")
    
    # Show global causal order
    lines.append("\nGlobal Causal Order:")
    lines.append("-" * 30)
    global_order = cluster.get_global_causal_order()
    for i, event in enumerate(global_order):
        lines.append(f"  {i+1}. {event.node_id}: {event.operation} {event.key}")
    
    return "\n".join(lines)


def trace_causality(event: Event, cluster: Cluster) -> List[Event]:
    """
    Trace the causal history of an event.
    
    Returns all events that causally precede the given event.
    """
    causal_predecessors = []
    
    for node in cluster.nodes.values():
        for e in node.events:
            if e.vector_clock.happens_before(event.vector_clock):
                causal_predecessors.append(e)
    
    # Sort by causal order
    causal_predecessors.sort(
        key=lambda e: sum(e.vector_clock.clock.values())
    )
    
    return causal_predecessors


def find_concurrent_groups(events: List[Event]) -> List[Set[int]]:
    """
    Find groups of mutually concurrent events.
    
    Returns a list of sets, where each set contains indices of
    events that are all concurrent with each other.
    """
    n = len(events)
    concurrent_graph = defaultdict(set)
    
    # Build concurrency graph
    for i in range(n):
        for j in range(i + 1, n):
            if events[i].vector_clock.is_concurrent_with(events[j].vector_clock):
                concurrent_graph[i].add(j)
                concurrent_graph[j].add(i)
    
    # Find cliques (groups where everyone is concurrent with everyone)
    groups = []
    visited = set()
    
    for i in range(n):
        if i in visited:
            continue
        
        # Find all nodes concurrent with i
        group = {i}
        candidates = concurrent_graph[i].copy()
        
        for j in candidates:
            if j in visited:
                continue
            # Check if j is concurrent with everyone in the group
            if all(j in concurrent_graph[k] or j == k for k in group):
                group.add(j)
        
        if len(group) > 1:
            groups.append(group)
            visited.update(group)
    
    return groups


def visualize_causality_graph(events: List[Event]) -> str:
    """
    Create a DOT graph representation of causality relationships.
    
    This can be rendered with Graphviz.
    """
    lines = ["digraph Causality {"]
    lines.append("  rankdir=TB;")
    lines.append("  node [shape=box];")
    
    # Create nodes
    for i, event in enumerate(events):
        label = f"{event.node_id}\\n{event.operation} {event.key}"
        lines.append(f'  E{i} [label="{label}"];')
    
    # Create edges for happens-before relationships
    for i, event_i in enumerate(events):
        for j, event_j in enumerate(events):
            if i != j and event_i.vector_clock.happens_before(event_j.vector_clock):
                # Check if there's a direct edge (no intermediate event)
                is_direct = True
                for k, event_k in enumerate(events):
                    if k != i and k != j:
                        if (event_i.vector_clock.happens_before(event_k.vector_clock) and
                            event_k.vector_clock.happens_before(event_j.vector_clock)):
                            is_direct = False
                            break
                if is_direct:
                    lines.append(f"  E{i} -> E{j};")
    
    # Create dashed edges for concurrency
    for i, event_i in enumerate(events):
        for j, event_j in enumerate(events):
            if i < j and event_i.vector_clock.is_concurrent_with(event_j.vector_clock):
                lines.append(f"  E{i} -> E{j} [style=dashed, color=red, dir=none];")
    
    lines.append("}")
    return "\n".join(lines)


def compare_clocks_detailed(vc1: VectorClock, vc2: VectorClock) -> str:
    """
    Create a detailed comparison of two vector clocks.
    """
    lines = []
    lines.append(f"Clock 1: {format_vector_clock(vc1)}")
    lines.append(f"Clock 2: {format_vector_clock(vc2)}")
    lines.append("")
    
    relation = vc1.compare(vc2)
    lines.append(f"Relationship: {relation.name}")
    lines.append("")
    
    # Component-wise comparison
    all_nodes = vc1.nodes() | vc2.nodes()
    lines.append("Component-wise comparison:")
    for node in sorted(all_nodes):
        v1 = vc1.get_timestamp(node)
        v2 = vc2.get_timestamp(node)
        if v1 < v2:
            symbol = "<"
        elif v1 > v2:
            symbol = ">"
        else:
            symbol = "="
        lines.append(f"  {node}: {v1} {symbol} {v2}")
    
    return "\n".join(lines)


def detect_causality_violations(events: List[Event]) -> List[Tuple[Event, Event, str]]:
    """
    Detect potential causality violations in a sequence of events.
    
    Returns a list of (event1, event2, reason) tuples for violations.
    """
    violations = []
    
    for i, event_i in enumerate(events):
        for j, event_j in enumerate(events[i+1:], start=i+1):
            # Check if j appears after i but i happens after j
            if event_i.vector_clock.happens_after(event_j.vector_clock):
                violations.append((
                    event_i,
                    event_j,
                    f"Event at position {i} happens after event at position {j}"
                ))
    
    return violations


# Example/demo functions
def demo_two_node_scenario():
    """
    Run a demo scenario with two nodes and print visualizations.
    """
    print("=" * 60)
    print("Demo: Two Node Scenario")
    print("=" * 60)
    
    # Create nodes
    node_a = Node("Alice")
    node_b = Node("Bob")
    
    print("\n1. Alice writes x=1")
    event1 = node_a.write("x", 1)
    print(f"   Alice's clock: {format_vector_clock(node_a.vector_clock)}")
    
    print("\n2. Bob writes y=2")
    event2 = node_b.write("y", 2)
    print(f"   Bob's clock: {format_vector_clock(node_b.vector_clock)}")
    
    print("\n3. Alice sends message to Bob")
    node_a.send_message(node_b, "x")
    print(f"   Alice's clock: {format_vector_clock(node_a.vector_clock)}")
    print(f"   Bob's clock: {format_vector_clock(node_b.vector_clock)}")
    
    print("\n4. Bob writes z=3 (after receiving from Alice)")
    event3 = node_b.write("z", 3)
    print(f"   Bob's clock: {format_vector_clock(node_b.vector_clock)}")
    
    print("\n" + "=" * 60)
    print("Causality Analysis")
    print("=" * 60)
    
    events = [event1, event2, event3]
    print("\nCausality Matrix:")
    print_causality_matrix(events)
    
    print("\nDetailed Comparisons:")
    for i, e1 in enumerate(events):
        for e2 in events[i+1:]:
            print(f"\nEvent {i+1} ({e1.node_id}) vs Event {i+2} ({e2.node_id}):")
            print(compare_clocks_detailed(e1.vector_clock, e2.vector_clock))


def demo_partition_scenario():
    """
    Run a demo scenario with network partition.
    """
    print("\n" + "=" * 60)
    print("Demo: Network Partition Scenario")
    print("=" * 60)
    
    cluster = Cluster()
    node_a = cluster.add_node("Node-A")
    node_b = cluster.add_node("Node-B")
    node_c = cluster.add_node("Node-C")
    
    print("\n1. Initial sync between all nodes")
    node_a.send_message(node_b, "init")
    node_b.send_message(node_c, "init")
    
    print("\n2. NETWORK PARTITION: A is isolated from B and C")
    print("   (Simulating partition by not sending messages)")
    
    print("\n3. During partition:")
    print("   - Node A writes x=1")
    event_a = node_a.write("x", 1)
    print(f"     Clock: {format_vector_clock(node_a.vector_clock)}")
    
    print("   - Node B writes x=2")
    event_b = node_b.write("x", 2)
    print(f"     Clock: {format_vector_clock(node_b.vector_clock)}")
    
    print("\n4. These writes are CONCURRENT (conflict!)")
    print(compare_clocks_detailed(event_a.vector_clock, event_b.vector_clock))
    
    print("\n5. Partition heals - B receives A's message")
    node_a.send_message(node_b, "x")
    
    print("\n6. Conflict detected:")
    conflicts = cluster.find_concurrent_conflicts()
    for e1, e2 in conflicts:
        print(f"   - {e1.node_id} and {e2.node_id} both wrote to '{e1.key}'")
    
    print("\n" + visualize_cluster_state(cluster))


if __name__ == "__main__":
    demo_two_node_scenario()
    demo_partition_scenario()
