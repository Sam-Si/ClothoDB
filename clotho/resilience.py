"""
Resilience mechanisms for handling edge cases in distributed systems.

This module implements:
1. Hinted Handoff - Handle temporary node failures
2. Read Repair - Ensure consistency across replicas
3. Circuit Breakers - Prevent cascading failures
4. Request retries with exponential backoff
"""

from __future__ import annotations

import asyncio
import time
import random
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from .vector_clock import VectorClock
from .consistent_hash import NodeInfo, Router


logger = logging.getLogger(__name__)


# ============================================================================
# 1. HINTED HANDOFF - Handle Temporary Node Failures
# ============================================================================

@dataclass
class Hint:
    """A hint for a write that needs to be forwarded to a target node."""
    target_node_id: str
    key: str
    value: Any
    vector_clock: VectorClock
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 5
    
    def should_retry(self) -> bool:
        """Check if this hint should be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry counter."""
        self.retry_count += 1


class HintedHandoffManager:
    """
    Manages hinted writes for temporarily unavailable nodes.
    
    When a write targets a node that is down, we store a "hint" on another
    node. When the target comes back up, we forward the hinted writes.
    
    This is a key DynamoDB technique for handling temporary failures.
    """
    
    HINT_EXPIRY_SECONDS = 3600  # 1 hour
    FORWARD_INTERVAL = 30  # Try to forward hints every 30 seconds
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.hints: Dict[str, List[Hint]] = defaultdict(list)
        self.running = False
        self._forward_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the background hint forwarding task."""
        self.running = True
        self._forward_task = asyncio.create_task(self._forward_loop())
        logger.info(f"HintedHandoffManager started for {self.node_id}")
    
    async def stop(self):
        """Stop the background task."""
        self.running = False
        if self._forward_task:
            self._forward_task.cancel()
            try:
                await self._forward_task
            except asyncio.CancelledError:
                pass
    
    def store_hint(
        self,
        target_node_id: str,
        key: str,
        value: Any,
        vector_clock: VectorClock
    ) -> None:
        """
        Store a hint for later forwarding.
        
        This is called when we can't reach the target node during a write.
        """
        hint = Hint(
            target_node_id=target_node_id,
            key=key,
            value=value,
            vector_clock=vector_clock
        )
        self.hints[target_node_id].append(hint)
        logger.info(f"Stored hint for {target_node_id}, key={key}")
    
    def get_hints_for_node(self, target_node_id: str) -> List[Hint]:
        """Get all pending hints for a specific target node."""
        return self.hints.get(target_node_id, [])
    
    def remove_hints_for_node(self, target_node_id: str) -> None:
        """Remove all hints for a node (after successful forwarding)."""
        if target_node_id in self.hints:
            del self.hints[target_node_id]
    
    def remove_hint(self, target_node_id: str, hint: Hint) -> None:
        """Remove a specific hint."""
        if target_node_id in self.hints:
            self.hints[target_node_id] = [
                h for h in self.hints[target_node_id]
                if h.timestamp != hint.timestamp or h.key != hint.key
            ]
    
    def _is_expired(self, hint: Hint) -> bool:
        """Check if a hint has expired."""
        return time.time() - hint.timestamp > self.HINT_EXPIRY_SECONDS
    
    def cleanup_expired_hints(self) -> int:
        """Remove expired hints and return count cleaned."""
        cleaned = 0
        for node_id in list(self.hints.keys()):
            original_count = len(self.hints[node_id])
            self.hints[node_id] = [
                h for h in self.hints[node_id]
                if not self._is_expired(h)
            ]
            cleaned += original_count - len(self.hints[node_id])
            if not self.hints[node_id]:
                del self.hints[node_id]
        return cleaned
    
    async def _forward_loop(self):
        """Background task to periodically try forwarding hints."""
        while self.running:
            try:
                await asyncio.sleep(self.FORWARD_INTERVAL)
                await self._try_forward_all_hints()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in forward loop: {e}")
    
    async def _try_forward_all_hints(self):
        """Try to forward all pending hints."""
        # Clean up expired hints first
        cleaned = self.cleanup_expired_hints()
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired hints")
        
        # Try to forward hints for each node
        for target_node_id in list(self.hints.keys()):
            await self._forward_hints_to_node(target_node_id)
    
    async def _forward_hints_to_node(self, target_node_id: str):
        """Forward all hints to a specific node."""
        hints = self.hints.get(target_node_id, [])
        if not hints:
            return
        
        logger.info(f"Attempting to forward {len(hints)} hints to {target_node_id}")
        
        # In a real implementation, this would make RPC calls
        # For now, we simulate success/failure
        successfully_forwarded = []
        
        for hint in hints:
            if not hint.should_retry():
                logger.warning(f"Hint for {hint.key} exceeded max retries, dropping")
                successfully_forwarded.append(hint)
                continue
            
            try:
                # Simulate forwarding - in reality this would be an RPC
                success = await self._send_hint_to_node(hint, target_node_id)
                
                if success:
                    successfully_forwarded.append(hint)
                    logger.info(f"Successfully forwarded hint for {hint.key}")
                else:
                    hint.increment_retry()
                    logger.warning(f"Failed to forward hint for {hint.key}, retry {hint.retry_count}")
            except Exception as e:
                hint.increment_retry()
                logger.error(f"Error forwarding hint for {hint.key}: {e}")
        
        # Remove successfully forwarded hints
        for hint in successfully_forwarded:
            self.remove_hint(target_node_id, hint)
    
    async def _send_hint_to_node(self, hint: Hint, target_node_id: str) -> bool:
        """
        Send a hinted write to the target node.
        
        Returns True if successful, False otherwise.
        """
        # This would make an actual RPC call in production
        # For simulation, we assume it works
        await asyncio.sleep(0.001)  # Simulate network delay
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about pending hints."""
        total_hints = sum(len(hints) for hints in self.hints.values())
        return {
            "total_pending_hints": total_hints,
            "nodes_with_hints": len(self.hints),
            "hints_by_node": {
                node_id: len(hints)
                for node_id, hints in self.hints.items()
            }
        }


# ============================================================================
# 2. READ REPAIR - Ensure Consistency Across Replicas
# ============================================================================

@dataclass
class ReadResult:
    """Result from reading a value from a node."""
    key: str
    value: Optional[Any]
    vector_clock: Optional[VectorClock]
    node_id: str
    found: bool
    error: Optional[str] = None


class ReadRepairHandler:
    """
    Handles reading from multiple replicas and repairing inconsistencies.
    
    DynamoDB reads from R replicas (R < N), compares vector clocks,
    and if divergent, reconciles and writes back the merged version.
    """
    
    def __init__(self, router: Router):
        self.router = router
        self.repair_count = 0
        self.conflict_count = 0
    
    async def read_with_repair(
        self,
        key: str,
        r: int = 2,
        timeout_ms: float = 100
    ) -> ReadResult:
        """
        Read from R replicas and repair any inconsistencies.
        
        Args:
            key: The key to read
            r: Number of replicas to read from
            timeout_ms: Timeout for each read
        
        Returns:
            ReadResult with the reconciled value
        """
        # Get preference list (top N nodes for this key)
        nodes = self.router.get_replica_nodes(key)
        
        if not nodes:
            return ReadResult(
                key=key,
                value=None,
                vector_clock=None,
                node_id="",
                found=False,
                error="No nodes available"
            )
        
        # Read from first R nodes
        read_tasks = [
            self._read_from_node(node, key, timeout_ms)
            for node in nodes[:r]
        ]
        
        # Wait for all reads to complete
        results = await asyncio.gather(*read_tasks, return_exceptions=True)
        
        # Filter out errors and not-found
        successful_reads = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Read error: {result}")
                continue
            if result.found and result.vector_clock is not None:
                successful_reads.append(result)
        
        if not successful_reads:
            return ReadResult(
                key=key,
                value=None,
                vector_clock=None,
                node_id=nodes[0].node_id if nodes else "",
                found=False
            )
        
        # Check for conflicts
        if len(successful_reads) == 1:
            return successful_reads[0]
        
        # Multiple versions - check for conflicts
        versions = [
            (r.value, r.vector_clock, r.node_id)
            for r in successful_reads
        ]
        
        conflict_exists = self._check_for_conflicts(versions)
        
        if conflict_exists:
            self.conflict_count += 1
            logger.info(f"Conflict detected for key={key}, reconciling...")
            
            # Reconcile versions
            reconciled_value, reconciled_clock = self._reconcile_versions(versions)
            
            # Trigger async read repair
            asyncio.create_task(
                self._write_repaired_value(key, reconciled_value, reconciled_clock, nodes)
            )
            
            return ReadResult(
                key=key,
                value=reconciled_value,
                vector_clock=reconciled_clock,
                node_id=successful_reads[0].node_id,
                found=True
            )
        else:
            # All versions agree, return first
            return successful_reads[0]
    
    async def _read_from_node(
        self,
        node: NodeInfo,
        key: str,
        timeout_ms: float
    ) -> ReadResult:
        """Read a value from a specific node."""
        try:
            # In production, this would be an RPC call
            # For now, simulate with a delay
            await asyncio.sleep(timeout_ms / 1000.0 * random.uniform(0.5, 1.5))
            
            # Simulate reading from node's storage
            # This would actually call the node's storage
            return ReadResult(
                key=key,
                value=None,  # Would be actual value
                vector_clock=None,  # Would be actual clock
                node_id=node.node_id,
                found=False  # Would be True if found
            )
        except asyncio.TimeoutError:
            return ReadResult(
                key=key,
                value=None,
                vector_clock=None,
                node_id=node.node_id,
                found=False,
                error="Timeout"
            )
    
    def _check_for_conflicts(
        self,
        versions: List[tuple]
    ) -> bool:
        """
        Check if any versions conflict (are concurrent).
        
        Returns True if there's a conflict that needs reconciliation.
        """
        if len(versions) < 2:
            return False
        
        # Compare all pairs
        for i in range(len(versions)):
            for j in range(i + 1, len(versions)):
                _, clock_i, _ = versions[i]
                _, clock_j, _ = versions[j]
                
                if clock_i and clock_j:
                    if clock_i.is_concurrent_with(clock_j):
                        return True
        
        return False
    
    def _reconcile_versions(
        self,
        versions: List[tuple]
    ) -> tuple:
        """
        Reconcile multiple versions into a single value.
        
        Strategy: Merge vector clocks and return all values (siblings)
        for client-side resolution.
        """
        # Merge all vector clocks
        merged_clock = versions[0][1]
        for _, clock, _ in versions[1:]:
            if clock:
                merged_clock = merged_clock.merge(clock)
        
        # Collect all unique values
        all_values = [v for v, _, _ in versions if v is not None]
        
        # If all values are the same, return single value
        if len(set(str(v) for v in all_values)) == 1:
            return all_values[0], merged_clock
        
        # Return siblings for client-side resolution
        siblings = {
            "values": all_values,
            "conflict": True,
            "needs_resolution": True
        }
        
        return siblings, merged_clock
    
    async def _write_repaired_value(
        self,
        key: str,
        value: Any,
        vector_clock: VectorClock,
        nodes: List[NodeInfo]
    ):
        """
        Write the reconciled value back to all replicas.
        
        This is the "read repair" part - we fix the inconsistent replicas.
        """
        logger.info(f"Starting read repair for key={key}")
        
        repair_tasks = [
            self._write_to_node(node, key, value, vector_clock)
            for node in nodes
        ]
        
        results = await asyncio.gather(*repair_tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        self.repair_count += 1
        
        logger.info(f"Read repair completed for key={key}: {success_count}/{len(nodes)} nodes updated")
    
    async def _write_to_node(
        self,
        node: NodeInfo,
        key: str,
        value: Any,
        vector_clock: VectorClock
    ):
        """Write a value to a specific node."""
        # In production, this would be an RPC call
        await asyncio.sleep(0.001)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get read repair statistics."""
        return {
            "repairs_initiated": self.repair_count,
            "conflicts_detected": self.conflict_count
        }


# ============================================================================
# 3. CIRCUIT BREAKER - Prevent Cascading Failures
# ============================================================================

class CircuitState(Enum):
    """States for circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    When a node fails repeatedly, we "open" the circuit and stop
    sending requests to it for a cooldown period.
    """
    
    def __init__(
        self,
        node_id: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        self.node_id = node_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute an operation with circuit breaker protection.
        
        Raises CircuitOpenError if circuit is open.
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit for {self.node_id} entering half-open state")
                else:
                    raise CircuitOpenError(f"Circuit open for {self.node_id}")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError(f"Circuit half-open limit reached for {self.node_id}")
                self.half_open_calls += 1
        
        # Execute the operation
        try:
            result = await operation(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Record a successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.half_open_max_calls:
                    self._reset()
                    logger.info(f"Circuit for {self.node_id} closed (recovered)")
            else:
                self.failure_count = 0
    
    async def _on_failure(self):
        """Record a failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit for {self.node_id} opened (failed in half-open)")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit for {self.node_id} opened ({self.failure_count} failures)")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _reset(self):
        """Reset the circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """Manages circuit breakers for multiple nodes."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, node_id: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a node."""
        if node_id not in self.breakers:
            self.breakers[node_id] = CircuitBreaker(node_id)
        return self.breakers[node_id]
    
    def remove_breaker(self, node_id: str):
        """Remove a circuit breaker."""
        if node_id in self.breakers:
            del self.breakers[node_id]
    
    def get_all_states(self) -> Dict[str, str]:
        """Get states of all circuit breakers."""
        return {
            node_id: breaker.get_state().value
            for node_id, breaker in self.breakers.items()
        }


# ============================================================================
# 4. RETRY POLICY - Exponential Backoff with Jitter
# ============================================================================

@dataclass
class RetryPolicy:
    """Configurable retry policy."""
    max_retries: int = 3
    base_delay: float = 0.01  # 10ms
    max_delay: float = 1.0    # 1s
    exponential_base: float = 2.0
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt."""
        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.jitter:
            delay = delay * random.uniform(0.5, 1.5)
        
        return delay


class ResilientOperation:
    """Execute operations with retry logic."""
    
    def __init__(
        self,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_manager: Optional[CircuitBreakerManager] = None
    ):
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_manager = circuit_manager
    
    async def execute(
        self,
        operation: Callable,
        node_id: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with retries and circuit breaker.
        
        Args:
            operation: The async function to execute
            node_id: Target node (for circuit breaker)
            *args, **kwargs: Arguments for operation
        
        Returns:
            Result of operation
        
        Raises:
            MaxRetriesExceeded: If all retries fail
            CircuitOpenError: If circuit is open
        """
        circuit = None
        if self.circuit_manager:
            circuit = self.circuit_manager.get_breaker(node_id)
        
        last_error = None
        
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                if circuit:
                    return await circuit.call(operation, *args, **kwargs)
                else:
                    return await operation(*args, **kwargs)
                    
            except (asyncio.TimeoutError, ConnectionError) as e:
                last_error = e
                
                if attempt < self.retry_policy.max_retries:
                    delay = self.retry_policy.get_delay(attempt)
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}), "
                        f"retrying in {delay:.3f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Operation failed after {attempt + 1} attempts")
                    raise MaxRetriesExceeded(f"Failed after {attempt + 1} attempts") from e
            
            except CircuitOpenError:
                raise  # Don't retry if circuit is open
            
            except Exception as e:
                # Don't retry on non-transient errors
                raise
        
        raise MaxRetriesExceeded("Unexpected end of retry loop") from last_error


class MaxRetriesExceeded(Exception):
    """Raised when max retries are exceeded."""
    pass


# ============================================================================
# 5. SLOPPY QUORUM - Availability During Partitions
# ============================================================================

class QuorumType(Enum):
    """Types of quorum."""
    STRICT = "strict"      # Must write to designated N nodes
    SLOPPY = "sloppy"      # Can write to any available N nodes


class QuorumManager:
    """
    Manages quorum-based reads and writes.
    
    DynamoDB uses sloppy quorum for availability during partitions:
    - Write to any W available nodes (not necessarily the designated ones)
    - Read from any R available nodes
    - Reconcile later when partition heals
    """
    
    def __init__(
        self,
        router: Router,
        n: int = 3,  # Replication factor
        w: int = 2,  # Write quorum
        r: int = 2,  # Read quorum
        quorum_type: QuorumType = QuorumType.SLOPPY
    ):
        self.router = router
        self.N = n
        self.W = w
        self.R = r
        self.quorum_type = quorum_type
    
    def get_write_nodes(self, key: str, available_nodes: Set[str]) -> List[NodeInfo]:
        """
        Get the list of nodes to write to.
        
        For strict quorum: must be in preference list
        For sloppy quorum: any available nodes
        """
        preference_list = self.router.get_replica_nodes(key)
        
        if self.quorum_type == QuorumType.STRICT:
            # Only use nodes in preference list
            writable = [n for n in preference_list if n.node_id in available_nodes]
        else:
            # Sloppy quorum: use any available nodes, prioritizing preference list
            writable = []
            
            # First, try preference list nodes
            for node in preference_list:
                if node.node_id in available_nodes:
                    writable.append(node)
            
            # If not enough, use any other available nodes
            if len(writable) < self.W:
                all_servers = self.router.get_all_servers()
                for server in all_servers:
                    if server.node_id in available_nodes and server not in writable:
                        writable.append(server)
                        if len(writable) >= self.W:
                            break
        
        return writable[:self.N]  # Never exceed N
    
    def get_read_nodes(self, key: str, available_nodes: Set[str]) -> List[NodeInfo]:
        """Get the list of nodes to read from."""
        preference_list = self.router.get_replica_nodes(key)
        readable = [n for n in preference_list if n.node_id in available_nodes]
        return readable
    
    def is_write_quorum_possible(self, available_nodes: Set[str]) -> bool:
        """Check if a write quorum is possible."""
        return len(available_nodes) >= self.W
    
    def is_read_quorum_possible(self, available_nodes: Set[str]) -> bool:
        """Check if a read quorum is possible."""
        return len(available_nodes) >= self.R


# ============================================================================
# 6. VECTOR CLOCK PRUNING - Prevent Unbounded Growth
# ============================================================================

class VectorClockPruner:
    """
    Prunes vector clocks to prevent unbounded growth.
    
    DynamoDB prunes clocks when they exceed ~10 entries.
    Falls back to timestamp-based resolution for pruned entries.
    """
    
    MAX_CLOCK_SIZE = 10
    
    def prune(self, clock: VectorClock) -> VectorClock:
        """
        Prune a vector clock if it exceeds MAX_CLOCK_SIZE.
        
        Strategy: Keep the entries with highest timestamps (most recent).
        """
        if len(clock) <= self.MAX_CLOCK_SIZE:
            return clock
        
        # Sort by timestamp and keep newest
        sorted_items = sorted(
            clock.clock.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        pruned_dict = dict(sorted_items[:self.MAX_CLOCK_SIZE])
        
        logger.warning(
            f"Pruned vector clock from {len(clock)} to {len(pruned_dict)} entries"
        )
        
        return VectorClock(clock=pruned_dict)
    
    def should_prune(self, clock: VectorClock) -> bool:
        """Check if a clock needs pruning."""
        return len(clock) > self.MAX_CLOCK_SIZE


# ============================================================================
# Integration: Resilient Distributed Storage
# ============================================================================

class ResilientStorage:
    """
    High-level storage interface combining all resilience mechanisms.
    
    This is what developers interact with - it handles all the edge cases
    internally.
    """
    
    def __init__(
        self,
        router: Router,
        node_id: str,
        hinted_handoff: Optional[HintedHandoffManager] = None,
        read_repair: Optional[ReadRepairHandler] = None,
        circuit_manager: Optional[CircuitBreakerManager] = None,
        quorum_manager: Optional[QuorumManager] = None,
        clock_pruner: Optional[VectorClockPruner] = None
    ):
        self.router = router
        self.node_id = node_id
        self.hinted_handoff = hinted_handoff or HintedHandoffManager(node_id)
        self.read_repair = read_repair or ReadRepairHandler(router)
        self.circuit_manager = circuit_manager or CircuitBreakerManager()
        self.quorum_manager = quorum_manager or QuorumManager(router)
        self.clock_pruner = clock_pruner or VectorClockPruner()
        self.resilient_op = ResilientOperation(circuit_manager=circuit_manager)
    
    async def write(
        self,
        key: str,
        value: Any,
        context: Optional[VectorClock] = None
    ) -> Dict[str, Any]:
        """
        Write a value with all resilience mechanisms.
        
        1. Get write nodes (using sloppy quorum if needed)
        2. Try to write to W nodes
        3. Store hints for failed nodes
        4. Return success if W writes succeeded
        """
        # Get nodes to write to
        available_nodes = self._get_available_nodes()
        write_nodes = self.quorum_manager.get_write_nodes(key, available_nodes)
        
        if len(write_nodes) < self.quorum_manager.W:
            return {
                "success": False,
                "error": f"Write quorum not possible (need {self.quorum_manager.W}, have {len(write_nodes)})"
            }
        
        # Prepare vector clock
        if context:
            clock = context.increment(self.node_id)
        else:
            clock = VectorClock.new(self.node_id).increment(self.node_id)
        
        # Prune if needed
        clock = self.clock_pruner.prune(clock)
        
        # Write to nodes
        successful_writes = 0
        failed_nodes = []
        
        for node in write_nodes:
            try:
                await self.resilient_op.execute(
                    self._write_to_node,
                    node.node_id,
                    node,
                    key,
                    value,
                    clock
                )
                successful_writes += 1
            except Exception as e:
                logger.warning(f"Write to {node.node_id} failed: {e}")
                failed_nodes.append(node)
        
        # Store hints for failed nodes
        for node in failed_nodes:
            self.hinted_handoff.store_hint(node.node_id, key, value, clock)
        
        # Check if we met quorum
        if successful_writes >= self.quorum_manager.W:
            return {
                "success": True,
                "key": key,
                "vector_clock": clock.to_dict(),
                "writes_succeeded": successful_writes,
                "writes_failed": len(failed_nodes),
                "hints_stored": len(failed_nodes)
            }
        else:
            return {
                "success": False,
                "error": f"Write quorum not met ({successful_writes}/{self.quorum_manager.W})",
                "writes_succeeded": successful_writes,
                "writes_failed": len(failed_nodes)
            }
    
    async def read(self, key: str) -> Dict[str, Any]:
        """
        Read a value with read repair.
        
        1. Read from R replicas
        2. Compare versions
        3. Repair if divergent
        4. Return reconciled value
        """
        result = await self.read_repair.read_with_repair(
            key,
            r=self.quorum_manager.R
        )
        
        return {
            "found": result.found,
            "key": result.key,
            "value": result.value,
            "vector_clock": result.vector_clock.to_dict() if result.vector_clock else None,
            "node_id": result.node_id
        }
    
    async def start(self):
        """Start background tasks."""
        await self.hinted_handoff.start()
    
    async def stop(self):
        """Stop background tasks."""
        await self.hinted_handoff.stop()
    
    def _get_available_nodes(self) -> Set[str]:
        """Get set of currently available node IDs."""
        # In production, this would check health/liveness
        # For now, assume all nodes in router are available
        return {n.node_id for n in self.router.get_all_servers()}
    
    async def _write_to_node(
        self,
        node: NodeInfo,
        key: str,
        value: Any,
        clock: VectorClock
    ):
        """Internal: Write to a specific node."""
        # This would be an RPC call in production
        await asyncio.sleep(0.001)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "hinted_handoff": self.hinted_handoff.get_stats(),
            "read_repair": self.read_repair.get_stats(),
            "circuit_breakers": self.circuit_manager.get_all_states()
        }
