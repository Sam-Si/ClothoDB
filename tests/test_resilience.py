"""
Tests for resilience mechanisms and edge case handling.

These tests verify:
1. Hinted handoff for temporary node failures
2. Read repair for consistency
3. Circuit breakers for cascading failure prevention
4. Retry policies with exponential backoff
5. Sloppy quorum during partitions
6. Vector clock pruning
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

from clotho.resilience import (
    Hint,
    HintedHandoffManager,
    ReadRepairHandler,
    ReadResult,
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitOpenError,
    RetryPolicy,
    ResilientOperation,
    QuorumManager,
    QuorumType,
    VectorClockPruner,
    ResilientStorage,
    MaxRetriesExceeded
)
from clotho.vector_clock import VectorClock
from clotho.consistent_hash import NodeInfo, Router, create_5_server_cluster


class TestHint:
    """Tests for the Hint dataclass."""
    
    def test_hint_creation(self):
        """Should create hint with correct attributes."""
        clock = VectorClock.new("node-1").increment("node-1")
        hint = Hint(
            target_node_id="node-2",
            key="test-key",
            value="test-value",
            vector_clock=clock
        )
        
        assert hint.target_node_id == "node-2"
        assert hint.key == "test-key"
        assert hint.value == "test-value"
        assert hint.retry_count == 0
        assert hint.max_retries == 5
        assert time.time() - hint.timestamp < 1  # Created recently
    
    def test_should_retry_initially(self):
        """Should allow retries initially."""
        hint = Hint(
            target_node_id="node-2",
            key="test-key",
            value="test-value",
            vector_clock=VectorClock.new("node-1")
        )
        assert hint.should_retry() is True
    
    def test_should_not_retry_after_max(self):
        """Should not allow retries after max reached."""
        hint = Hint(
            target_node_id="node-2",
            key="test-key",
            value="test-value",
            vector_clock=VectorClock.new("node-1")
        )
        hint.retry_count = 5
        assert hint.should_retry() is False
    
    def test_increment_retry(self):
        """Should increment retry counter."""
        hint = Hint(
            target_node_id="node-2",
            key="test-key",
            value="test-value",
            vector_clock=VectorClock.new("node-1")
        )
        hint.increment_retry()
        assert hint.retry_count == 1


class TestHintedHandoffManager:
    """Tests for HintedHandoffManager."""
    
    @pytest.fixture
    def manager(self):
        return HintedHandoffManager("node-1")
    
    def test_store_hint(self, manager):
        """Should store hint for target node."""
        clock = VectorClock.new("node-1").increment("node-1")
        manager.store_hint("node-2", "key-1", "value-1", clock)
        
        hints = manager.get_hints_for_node("node-2")
        assert len(hints) == 1
        assert hints[0].key == "key-1"
        assert hints[0].value == "value-1"
    
    def test_store_multiple_hints_same_node(self, manager):
        """Should store multiple hints for same target."""
        clock = VectorClock.new("node-1").increment("node-1")
        manager.store_hint("node-2", "key-1", "value-1", clock)
        manager.store_hint("node-2", "key-2", "value-2", clock)
        
        hints = manager.get_hints_for_node("node-2")
        assert len(hints) == 2
    
    def test_store_hints_different_nodes(self, manager):
        """Should store hints for different targets separately."""
        clock = VectorClock.new("node-1").increment("node-1")
        manager.store_hint("node-2", "key-1", "value-1", clock)
        manager.store_hint("node-3", "key-2", "value-2", clock)
        
        assert len(manager.get_hints_for_node("node-2")) == 1
        assert len(manager.get_hints_for_node("node-3")) == 1
    
    def test_remove_hints_for_node(self, manager):
        """Should remove all hints for a node."""
        clock = VectorClock.new("node-1").increment("node-1")
        manager.store_hint("node-2", "key-1", "value-1", clock)
        manager.remove_hints_for_node("node-2")
        
        assert len(manager.get_hints_for_node("node-2")) == 0
    
    def test_cleanup_expired_hints(self, manager):
        """Should remove expired hints."""
        clock = VectorClock.new("node-1")
        
        # Create hint with old timestamp
        hint = Hint(
            target_node_id="node-2",
            key="key-1",
            value="value-1",
            vector_clock=clock,
            timestamp=time.time() - 7200  # 2 hours ago
        )
        manager.hints["node-2"].append(hint)
        
        cleaned = manager.cleanup_expired_hints()
        assert cleaned == 1
        assert len(manager.get_hints_for_node("node-2")) == 0
    
    def test_get_stats(self, manager):
        """Should return statistics."""
        clock = VectorClock.new("node-1").increment("node-1")
        manager.store_hint("node-2", "key-1", "value-1", clock)
        manager.store_hint("node-2", "key-2", "value-2", clock)
        manager.store_hint("node-3", "key-3", "value-3", clock)
        
        stats = manager.get_stats()
        assert stats["total_pending_hints"] == 3
        assert stats["nodes_with_hints"] == 2
        assert stats["hints_by_node"]["node-2"] == 2
        assert stats["hints_by_node"]["node-3"] == 1
    
    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        """Should start and stop background task."""
        await manager.start()
        assert manager.running is True
        assert manager._forward_task is not None
        
        await manager.stop()
        assert manager.running is False


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""
    
    @pytest.fixture
    def cb(self):
        return CircuitBreaker(
            "node-1",
            failure_threshold=3,
            recovery_timeout=0.1,
            half_open_max_calls=2
        )
    
    @pytest.mark.asyncio
    async def test_successful_call(self, cb):
        """Should execute operation successfully."""
        async def success_op():
            return "success"
        
        result = await cb.call(success_op)
        assert result == "success"
        assert cb.get_state().value == "closed"
    
    @pytest.mark.asyncio
    async def test_failure_count_increments(self, cb):
        """Should count failures."""
        async def fail_op():
            raise ConnectionError("Failed")
        
        try:
            await cb.call(fail_op)
        except ConnectionError:
            pass
        
        assert cb.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_opens_after_threshold(self, cb):
        """Should open circuit after threshold failures."""
        async def fail_op():
            raise ConnectionError("Failed")
        
        # Trigger 3 failures
        for _ in range(3):
            try:
                await cb.call(fail_op)
            except ConnectionError:
                pass
        
        assert cb.get_state().value == "open"
    
    @pytest.mark.asyncio
    async def test_rejects_when_open(self, cb):
        """Should reject calls when circuit is open."""
        from clotho.resilience import CircuitState
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.time()
        
        async def any_op():
            return "result"
        
        with pytest.raises(CircuitOpenError):
            await cb.call(any_op)
    
    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, cb):
        """Should enter half-open after recovery timeout."""
        from clotho.resilience import CircuitState
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.time() - 0.2  # Past timeout
        
        async def success_op():
            return "success"
        
        # First call should work in half-open
        result = await cb.call(success_op)
        assert result == "success"
        assert cb.get_state().value == "half_open"
    
    @pytest.mark.asyncio
    async def test_closes_after_half_open_success(self, cb):
        """Should close circuit after half-open successes."""
        from clotho.resilience import CircuitState
        cb.state = CircuitState.HALF_OPEN
        
        async def success_op():
            return "success"
        
        # Need 2 successes to close
        await cb.call(success_op)
        await cb.call(success_op)
        
        assert cb.get_state().value == "closed"


class TestCircuitBreakerManager:
    """Tests for CircuitBreakerManager."""
    
    @pytest.fixture
    def manager(self):
        return CircuitBreakerManager()
    
    def test_get_breaker_creates_new(self, manager):
        """Should create new breaker for unknown node."""
        cb = manager.get_breaker("node-1")
        assert cb.node_id == "node-1"
    
    def test_get_breaker_returns_existing(self, manager):
        """Should return existing breaker."""
        cb1 = manager.get_breaker("node-1")
        cb2 = manager.get_breaker("node-1")
        assert cb1 is cb2
    
    def test_remove_breaker(self, manager):
        """Should remove breaker."""
        manager.get_breaker("node-1")
        manager.remove_breaker("node-1")
        
        # Getting again creates new
        cb = manager.get_breaker("node-1")
        assert cb.failure_count == 0  # Fresh breaker
    
    def test_get_all_states(self, manager):
        """Should return states for all breakers."""
        manager.get_breaker("node-1")
        manager.get_breaker("node-2")
        
        states = manager.get_all_states()
        assert "node-1" in states
        assert "node-2" in states


class TestRetryPolicy:
    """Tests for RetryPolicy."""
    
    def test_get_delay_exponential(self):
        """Should use exponential backoff."""
        policy = RetryPolicy(base_delay=0.01, exponential_base=2.0, jitter=False)
        
        assert policy.get_delay(0) == 0.01  # 10ms
        assert policy.get_delay(1) == 0.02  # 20ms
        assert policy.get_delay(2) == 0.04  # 40ms
    
    def test_get_delay_respects_max(self):
        """Should not exceed max delay."""
        policy = RetryPolicy(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=10.0,
            jitter=False
        )
        
        delay = policy.get_delay(1)  # Would be 10s without max
        assert delay == 5.0
    
    def test_get_delay_with_jitter(self):
        """Should add jitter when enabled."""
        policy = RetryPolicy(base_delay=1.0, jitter=True)
        
        delays = [policy.get_delay(0) for _ in range(100)]
        # All should be within 50%-150% of base
        assert all(0.5 <= d <= 1.5 for d in delays)
        # Should have some variation
        assert len(set(delays)) > 50


class TestResilientOperation:
    """Tests for ResilientOperation."""
    
    @pytest.fixture
    def resilient_op(self):
        return ResilientOperation()
    
    @pytest.mark.asyncio
    async def test_success_on_first_try(self, resilient_op):
        """Should succeed on first attempt."""
        async def success_op():
            return "success"
        
        result = await resilient_op.execute(success_op, "node-1")
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, resilient_op):
        """Should retry on timeout."""
        call_count = 0
        
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError("Timeout")
            return "success"
        
        result = await resilient_op.execute(fail_then_succeed, "node-1")
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, resilient_op):
        """Should retry on connection error."""
        call_count = 0
        
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Failed")
            return "success"
        
        result = await resilient_op.execute(fail_then_succeed, "node-1")
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, resilient_op):
        """Should raise MaxRetriesExceeded after max retries."""
        async def always_fail():
            raise asyncio.TimeoutError("Always fails")
        
        with pytest.raises(MaxRetriesExceeded):
            await resilient_op.execute(always_fail, "node-1")
    
    @pytest.mark.asyncio
    async def test_no_retry_on_other_errors(self, resilient_op):
        """Should not retry on non-transient errors."""
        call_count = 0
        
        async def value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Bad input")
        
        with pytest.raises(ValueError):
            await resilient_op.execute(value_error, "node-1")
        
        assert call_count == 1  # No retry


class TestQuorumManager:
    """Tests for QuorumManager."""
    
    @pytest.fixture
    def router(self):
        return create_5_server_cluster()
    
    @pytest.fixture
    def quorum_manager(self, router):
        return QuorumManager(router, n=3, w=2, r=2)
    
    def test_strict_quorum_uses_preference_list(self, router):
        """Strict quorum should only use preference list nodes."""
        manager = QuorumManager(
            router, n=3, w=2, r=2, quorum_type=QuorumType.STRICT
        )
        
        # All nodes available
        available = {f"server-{i}" for i in range(1, 6)}
        nodes = manager.get_write_nodes("test-key", available)
        
        # Should return nodes from preference list
        preference = {n.node_id for n in router.get_replica_nodes("test-key")}
        for node in nodes:
            assert node.node_id in preference
    
    def test_sloppy_quorum_uses_any_available(self, router):
        """Sloppy quorum should use any available nodes if needed."""
        manager = QuorumManager(
            router, n=3, w=2, r=2, quorum_type=QuorumType.SLOPPY
        )
        
        # Only 1 preference list node available
        preference = router.get_replica_nodes("test-key")
        available = {preference[0].node_id, "server-4", "server-5"}
        
        nodes = manager.get_write_nodes("test-key", available)
        
        # Should still get 2 nodes for W=2
        assert len(nodes) >= 2
    
    def test_write_quorum_possible(self, quorum_manager):
        """Should detect when write quorum is possible."""
        assert quorum_manager.is_write_quorum_possible({"a", "b", "c"}) is True
        assert quorum_manager.is_write_quorum_possible({"a"}) is False
    
    def test_read_quorum_possible(self, quorum_manager):
        """Should detect when read quorum is possible."""
        assert quorum_manager.is_read_quorum_possible({"a", "b"}) is True
        assert quorum_manager.is_read_quorum_possible({"a"}) is False


class TestVectorClockPruner:
    """Tests for VectorClockPruner."""
    
    @pytest.fixture
    def pruner(self):
        return VectorClockPruner()
    
    def test_no_pruning_needed(self, pruner):
        """Should not prune small clocks."""
        clock = VectorClock.new("node-1").increment("node-1")
        pruned = pruner.prune(clock)
        
        assert pruned == clock
    
    def test_prunes_large_clock(self, pruner):
        """Should prune clocks exceeding max size."""
        # Create clock with 15 entries
        clock = VectorClock({f"node-{i}": i for i in range(15)})
        
        pruned = pruner.prune(clock)
        
        assert len(pruned) <= pruner.MAX_CLOCK_SIZE
    
    def test_keeps_highest_timestamps(self, pruner):
        """Should keep entries with highest timestamps."""
        # Create clock with entries 0-14
        clock = VectorClock({f"node-{i}": i for i in range(15)})
        
        pruned = pruner.prune(clock)
        
        # Should keep node-5 through node-14 (highest 10)
        for i in range(5, 15):
            assert f"node-{i}" in pruned.clock
        
        # Should drop node-0 through node-4
        for i in range(5):
            assert f"node-{i}" not in pruned.clock
    
    def test_should_prune_detects_large_clock(self, pruner):
        """Should correctly detect when pruning is needed."""
        small_clock = VectorClock({"a": 1, "b": 2})
        large_clock = VectorClock({f"node-{i}": i for i in range(15)})
        
        assert pruner.should_prune(small_clock) is False
        assert pruner.should_prune(large_clock) is True


class TestReadRepairHandler:
    """Tests for ReadRepairHandler."""
    
    @pytest.fixture
    def router(self):
        return create_5_server_cluster()
    
    @pytest.fixture
    def handler(self, router):
        return ReadRepairHandler(router)
    
    def test_check_for_conflicts_no_conflict(self, handler):
        """Should detect no conflict when versions agree."""
        clock1 = VectorClock.new("a").increment("a")
        clock2 = clock1.increment("b")  # B happens after A
        
        versions = [
            ("val1", clock1, "node-1"),
            ("val2", clock2, "node-2")
        ]
        
        assert handler._check_for_conflicts(versions) is False
    
    def test_check_for_conflicts_concurrent(self, handler):
        """Should detect conflict for concurrent versions."""
        clock1 = VectorClock.new("a").increment("a")
        clock2 = VectorClock.new("b").increment("b")
        
        versions = [
            ("val1", clock1, "node-1"),
            ("val2", clock2, "node-2")
        ]
        
        assert handler._check_for_conflicts(versions) is True
    
    def test_reconcile_versions_merges_clocks(self, handler):
        """Should merge vector clocks during reconciliation."""
        clock1 = VectorClock({"a": 1, "b": 0})
        clock2 = VectorClock({"a": 0, "b": 1})
        
        versions = [
            ("val1", clock1, "node-1"),
            ("val2", clock2, "node-2")
        ]
        
        merged_val, merged_clock = handler._reconcile_versions(versions)
        
        assert merged_clock.clock["a"] == 1
        assert merged_clock.clock["b"] == 1
    
    def test_reconcile_returns_siblings_for_different_values(self, handler):
        """Should return siblings when values differ."""
        clock1 = VectorClock.new("a").increment("a")
        clock2 = VectorClock.new("b").increment("b")
        
        versions = [
            ("val1", clock1, "node-1"),
            ("val2", clock2, "node-2")
        ]
        
        merged_val, _ = handler._reconcile_versions(versions)
        
        assert merged_val["conflict"] is True
        assert "val1" in merged_val["values"]
        assert "val2" in merged_val["values"]
    
    def test_get_stats(self, handler):
        """Should return statistics."""
        stats = handler.get_stats()
        assert "repairs_initiated" in stats
        assert "conflicts_detected" in stats


class TestResilientStorage:
    """Tests for ResilientStorage."""
    
    @pytest.fixture
    def router(self):
        return create_5_server_cluster()
    
    @pytest.fixture
    def storage(self, router):
        return ResilientStorage(router, "client-1")
    
    @pytest.mark.asyncio
    async def test_write_returns_success(self, storage):
        """Write should return success status."""
        result = await storage.write("key-1", {"data": "value"})
        
        assert "success" in result
        assert "key" in result
        assert "vector_clock" in result
    
    @pytest.mark.asyncio
    async def test_write_with_context(self, storage):
        """Write should use provided context clock."""
        context = VectorClock.new("client-1")
        
        result = await storage.write("key-1", {"data": "value"}, context)
        
        # Clock should be incremented
        clock_dict = result["vector_clock"]
        assert clock_dict.get("client-1", 0) >= 1
    
    @pytest.mark.asyncio
    async def test_read_returns_result(self, storage):
        """Read should return result structure."""
        result = await storage.read("key-1")
        
        assert "found" in result
        assert "key" in result
        assert "value" in result
        assert "vector_clock" in result
    
    def test_get_stats(self, storage):
        """Should return comprehensive stats."""
        stats = storage.get_stats()
        
        assert "hinted_handoff" in stats
        assert "read_repair" in stats
        assert "circuit_breakers" in stats
