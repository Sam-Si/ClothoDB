"""
Write Coordinator for ClothoDB.

Implements the hybrid write path:
- Quorum-based writes (N=3, W=2)
- WAL for durability
- Vector clocks for causality
- Hinted handoff for temporary failures
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

from .vector_clock import VectorClock
from .consistent_hash import Router, NodeInfo
from .storage import StorageEngine
from .resilience import (
    HintedHandoffManager,
    CircuitBreakerManager,
    RetryPolicy,
    ResilientOperation
)


@dataclass
class WriteResult:
    """Result of a write operation."""
    success: bool
    key: str
    vector_clock: Optional[Dict[str, int]]
    replicas_written: int
    replicas_failed: int
    hints_stored: int
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class WriteOperation:
    """Represents a write operation to be sent to replicas."""
    key: str
    value: Any
    vector_clock: VectorClock
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "vector_clock": self.vector_clock.to_dict(),
            "timestamp": self.timestamp
        }


class ReplicaClient:
    """
    Client for communicating with a replica node.
    
    In production, this would make RPC calls (HTTP/gRPC).
    For now, simulates network delay and may inject failures for testing.
    """
    
    def __init__(
        self,
        node_info: NodeInfo,
        storage: Optional[StorageEngine] = None,
        failure_rate: float = 0.0  # For testing: 0.0-1.0
    ):
        self.node_info = node_info
        self.storage = storage
        self.failure_rate = failure_rate
    
    async def write(
        self,
        operation: WriteOperation
    ) -> Dict[str, Any]:
        """
        Send write to this replica.
        
        In production: HTTP POST /write
        Here: Direct storage call with simulated latency
        """
        # Simulate network latency (1-5ms)
        await asyncio.sleep(0.001 + 0.004 * hash(operation.key) % 1)
        
        # Simulate random failures for testing
        if self.failure_rate > 0:
            import random
            if random.random() < self.failure_rate:
                raise ConnectionError(f"Simulated failure to {self.node_info.node_id}")
        
        if self.storage:
            # Direct write to local storage
            await self.storage.write(
                operation.key,
                operation.value,
                operation.vector_clock
            )
            return {
                "success": True,
                "node_id": self.node_info.node_id,
                "ack": True
            }
        else:
            # Would make RPC call in production
            return {"success": True, "node_id": self.node_info.node_id}


class WriteCoordinator:
    """
    Coordinates writes across multiple replicas.
    
    Write flow:
    1. Get N replicas from consistent hashing
    2. Prepare write with vector clock
    3. Send to all N replicas concurrently
    4. Wait for W acknowledgments (quorum)
    5. Store hints for failed replicas
    6. Return success if quorum reached
    """
    
    def __init__(
        self,
        router: Router,
        storage_engines: Dict[str, StorageEngine],
        n: int = 3,  # Replication factor
        w: int = 2,  # Write quorum
        hinted_handoff: Optional[HintedHandoffManager] = None,
        retry_policy: Optional[RetryPolicy] = None
    ):
        self.router = router
        self.storage_engines = storage_engines
        self.n = n
        self.w = w
        
        self.hinted_handoff = hinted_handoff or HintedHandoffManager("coordinator")
        self.circuit_manager = CircuitBreakerManager()
        self.retry_policy = retry_policy or RetryPolicy(
            max_retries=3,
            base_delay=0.01,
            max_delay=0.5,
            jitter=True
        )
        self.resilient_op = ResilientOperation(
            retry_policy=self.retry_policy,
            circuit_manager=self.circuit_manager
        )
        
        # Stats
        self.writes_total = 0
        self.writes_successful = 0
        self.writes_failed = 0
        self.hints_total = 0
    
    async def start(self):
        """Start the coordinator."""
        await self.hinted_handoff.start()
    
    async def stop(self):
        """Stop the coordinator."""
        await self.hinted_handoff.stop()
    
    async def write(
        self,
        key: str,
        value: Any,
        context: Optional[VectorClock] = None
    ) -> WriteResult:
        """
        Write a key-value pair with quorum consistency.
        
        Args:
            key: The key to write
            value: The value to store
            context: Optional vector clock from previous read (for causality)
        
        Returns:
            WriteResult with success status and metadata
        """
        start_time = time.time()
        self.writes_total += 1
        
        try:
            # 1. Get N replicas
            replicas = self.router.get_replica_nodes(key)
            
            if len(replicas) < self.w:
                return WriteResult(
                    success=False,
                    key=key,
                    vector_clock=None,
                    replicas_written=0,
                    replicas_failed=0,
                    hints_stored=0,
                    error=f"Not enough replicas: {len(replicas)} < {self.w}",
                    latency_ms=(time.time() - start_time) * 1000
                )
            
            # 2. Prepare vector clock
            if context:
                # Increment from provided context
                coordinator_id = "coordinator"  # Should be unique per instance
                new_clock = context.increment(coordinator_id)
            else:
                # New clock starting with first replica
                new_clock = VectorClock.new(replicas[0].node_id).increment(replicas[0].node_id)
            
            operation = WriteOperation(
                key=key,
                value=value,
                vector_clock=new_clock,
                timestamp=time.time()
            )
            
            # 3. Send to all N replicas concurrently
            write_tasks = []
            for replica in replicas[:self.n]:
                task = asyncio.create_task(
                    self._write_to_replica(replica, operation)
                )
                write_tasks.append(task)
            
            # 4. Wait for responses with timeout
            timeout = 5.0  # 5 second timeout
            done, pending = await asyncio.wait(
                write_tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # 5. Count successes and failures
            successes = []
            failures = []
            
            for task in done:
                try:
                    result = task.result()
                    if result.get("success"):
                        successes.append(result)
                    else:
                        failures.append((result.get("node_id"), "Write failed"))
                except Exception as e:
                    # Find which replica failed
                    failures.append(("unknown", str(e)))
            
            # 6. Store hints for failed replicas
            hints_stored = 0
            replica_list = replicas[:self.n]
            for i, task in enumerate(write_tasks):
                if i < len(replica_list):
                    try:
                        task.result()
                    except Exception:
                        failed_node = replica_list[i]
                        self.hinted_handoff.store_hint(
                            failed_node.node_id,
                            key,
                            value,
                            new_clock
                        )
                        hints_stored += 1
                        self.hints_total += 1
            
            # 7. Check if quorum reached
            if len(successes) >= self.w:
                self.writes_successful += 1
                return WriteResult(
                    success=True,
                    key=key,
                    vector_clock=new_clock.to_dict(),
                    replicas_written=len(successes),
                    replicas_failed=len(failures),
                    hints_stored=hints_stored,
                    latency_ms=(time.time() - start_time) * 1000
                )
            else:
                self.writes_failed += 1
                return WriteResult(
                    success=False,
                    key=key,
                    vector_clock=new_clock.to_dict(),
                    replicas_written=len(successes),
                    replicas_failed=len(failures),
                    hints_stored=hints_stored,
                    error=f"Quorum not reached: {len(successes)}/{self.w}",
                    latency_ms=(time.time() - start_time) * 1000
                )
        
        except Exception as e:
            self.writes_failed += 1
            return WriteResult(
                success=False,
                key=key,
                vector_clock=None,
                replicas_written=0,
                replicas_failed=0,
                hints_stored=0,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def _write_to_replica(
        self,
        replica: NodeInfo,
        operation: WriteOperation
    ) -> Dict[str, Any]:
        """
        Write to a specific replica with retry and circuit breaker.
        """
        async def do_write():
            # Get storage engine for this replica
            storage = self.storage_engines.get(replica.node_id)
            
            if storage is None:
                # Should not happen in production
                raise ValueError(f"No storage for replica {replica.node_id}")
            
            # Perform write
            await storage.write(
                operation.key,
                operation.value,
                operation.vector_clock
            )
            
            return {
                "success": True,
                "node_id": replica.node_id,
                "ack": True
            }
        
        # Use resilient operation (retry + circuit breaker)
        return await self.resilient_op.execute(do_write, replica.node_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get write coordinator statistics."""
        return {
            "writes_total": self.writes_total,
            "writes_successful": self.writes_successful,
            "writes_failed": self.writes_failed,
            "success_rate": (
                self.writes_successful / self.writes_total * 100
                if self.writes_total > 0 else 0
            ),
            "hints_total": self.hints_total,
            "hinted_handoff": self.hinted_handoff.get_stats(),
            "circuit_breakers": self.circuit_manager.get_all_states()
        }


class DistributedWriteClient:
    """
    High-level client for distributed writes.
    
    Combines WriteCoordinator with convenience methods.
    """
    
    def __init__(
        self,
        router: Router,
        data_dir: str = "./data",
        n: int = 3,
        w: int = 2
    ):
        self.router = router
        self.data_dir = data_dir
        self.n = n
        self.w = w
        
        # Create storage engine for each node
        self.storage_engines: Dict[str, StorageEngine] = {}
        for node in router.get_all_servers():
            engine = StorageEngine(
                data_dir=f"{data_dir}/{node.node_id}",
                memtable_size=10000,
                wal_group_commit_ms=10.0
            )
            self.storage_engines[node.node_id] = engine
        
        # Create coordinator
        self.coordinator = WriteCoordinator(
            router=router,
            storage_engines=self.storage_engines,
            n=n,
            w=w
        )
    
    async def start(self):
        """Start all storage engines and coordinator."""
        # Start all storage engines
        for engine in self.storage_engines.values():
            await engine.start()
        
        # Start coordinator
        await self.coordinator.start()
    
    async def stop(self):
        """Stop all components."""
        await self.coordinator.stop()
        
        for engine in self.storage_engines.values():
            await engine.stop()
    
    async def put(self, key: str, value: Any) -> WriteResult:
        """
        Store a key-value pair.
        
        Example:
            result = await client.put("user:123", {"name": "Alice"})
            if result.success:
                print(f"Written to {result.replicas_written} replicas")
        """
        return await self.coordinator.write(key, value)
    
    async def get(self, key: str) -> Optional[Tuple[Any, VectorClock]]:
        """
        Read a key from the appropriate storage engine.
        
        For now, reads from primary. In production, would read from R replicas.
        """
        primary = self.router.get_primary_node(key)
        if primary is None:
            return None
        
        engine = self.storage_engines.get(primary.node_id)
        if engine is None:
            return None
        
        return await engine.read(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        storage_stats = {
            node_id: engine.get_stats()
            for node_id, engine in self.storage_engines.items()
        }
        
        return {
            "coordinator": self.coordinator.get_stats(),
            "storage": storage_stats
        }
