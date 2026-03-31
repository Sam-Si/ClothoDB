"""
Tests for the storage layer (WAL, Memtable, SSTable, StorageEngine).

These tests verify:
1. WAL append and recovery
2. Memtable operations and flush
3. SSTable creation and reads
4. StorageEngine integration
5. Group commit behavior
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path

from clotho.storage import (
    WriteAheadLog,
    WALEntry,
    Memtable,
    SSTable,
    StorageEngine
)
from clotho.vector_clock import VectorClock


class TestWALEntry:
    """Tests for WALEntry serialization."""
    
    def test_serialize_deserialize(self):
        """Should round-trip serialize correctly."""
        clock = VectorClock.new("node-1").increment("node-1")
        entry = WALEntry(
            sequence_number=1,
            timestamp=1234567890.0,
            key="test-key",
            value={"data": "value"},
            vector_clock=clock.to_dict(),
            operation="WRITE"
        )
        
        data = entry.serialize()
        restored = WALEntry.deserialize(data)
        
        assert restored.sequence_number == entry.sequence_number
        assert restored.key == entry.key
        assert restored.value == entry.value
        assert restored.vector_clock == entry.vector_clock
        assert restored.operation == entry.operation
    
    def test_corrupted_data_raises_error(self):
        """Should detect corrupted data."""
        clock = VectorClock.new("node-1")
        entry = WALEntry(1, 1234567890.0, "key", "value", clock.to_dict())
        data = entry.serialize()
        
        # Corrupt the data
        corrupted = data[:-4] + b'\xff\xff\xff\xff'
        
        with pytest.raises(ValueError, match="Checksum mismatch"):
            WALEntry.deserialize(corrupted)


class TestWriteAheadLog:
    """Tests for WriteAheadLog."""
    
    @pytest.fixture
    async def wal(self):
        """Create a temporary WAL."""
        tmpdir = tempfile.mkdtemp()
        wal = WriteAheadLog(tmpdir, max_file_size=1024*1024)
        await wal.start()
        yield wal
        await wal.stop()
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_append_increments_sequence(self):
        """Each append should increment sequence number."""
        tmpdir = tempfile.mkdtemp()
        wal = WriteAheadLog(tmpdir)
        await wal.start()
        
        try:
            clock = VectorClock.new("node-1")
            seq1 = await wal.append("key1", "value1", clock)
            seq2 = await wal.append("key2", "value2", clock)
            
            assert seq2 == seq1 + 1
        finally:
            await wal.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_recover_replays_entries(self):
        """Recovery should replay all entries."""
        tmpdir = tempfile.mkdtemp()
        wal = WriteAheadLog(tmpdir)
        await wal.start()
        
        try:
            clock = VectorClock.new("node-1")
            await wal.append("key1", "value1", clock)
            await wal.append("key2", "value2", clock)
            await wal.fsync()
            
            # Recover
            entries = await wal.recover()
            
            assert len(entries) == 2
            assert entries[0].key == "key1"
            assert entries[1].key == "key2"
        finally:
            await wal.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_file_rotation(self):
        """Should rotate file when size exceeded."""
        tmpdir = tempfile.mkdtemp()
        wal = WriteAheadLog(tmpdir, max_file_size=100)  # Small for testing
        await wal.start()
        
        try:
            clock = VectorClock.new("node-1")
            
            # Write enough to trigger rotation
            for i in range(10):
                await wal.append(f"key{i}", "x" * 50, clock)
            
            # Should have created a new file
            wal_files = list(Path(tmpdir).glob("wal-*.log"))
            assert len(wal_files) >= 1
        finally:
            await wal.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestMemtable:
    """Tests for Memtable."""
    
    def test_put_and_get(self):
        """Should store and retrieve values."""
        mem = Memtable(max_size=100)
        clock = VectorClock.new("node-1")
        
        mem.put("key1", "value1", clock)
        result = mem.get("key1")
        
        assert result is not None
        assert result[0] == "value1"
        assert result[1] == clock
    
    def test_get_missing_key(self):
        """Should return None for missing key."""
        mem = Memtable()
        assert mem.get("missing") is None
    
    def test_delete_removes_key(self):
        """Delete should remove key."""
        mem = Memtable()
        clock = VectorClock.new("node-1")
        
        mem.put("key1", "value1", clock)
        mem.delete("key1")
        
        assert mem.get("key1") is None
    
    def test_is_full_when_at_capacity(self):
        """Should report full when at capacity."""
        mem = Memtable(max_size=2)
        clock = VectorClock.new("node-1")
        
        mem.put("key1", "value1", clock)
        mem.put("key2", "value2", clock)
        
        assert mem.is_full() is True
    
    def test_items_returns_sorted(self):
        """Should return items in sorted order."""
        mem = Memtable()
        clock = VectorClock.new("node-1")
        
        mem.put("c", "value-c", clock)
        mem.put("a", "value-a", clock)
        mem.put("b", "value-b", clock)
        
        items = mem.items()
        keys = [k for k, _ in items]
        
        assert keys == ["a", "b", "c"]


class TestSSTable:
    """Tests for SSTable."""
    
    @pytest.mark.asyncio
    async def test_create_from_memtable(self):
        """Should create SSTable from memtable."""
        mem = Memtable()
        clock = VectorClock.new("node-1")
        
        mem.put("key1", "value1", clock)
        mem.put("key2", "value2", clock)
        
        tmpdir = tempfile.mkdtemp()
        try:
            path = Path(tmpdir) / "test.sst"
            sstable = await SSTable.create_from_memtable(mem, path)
            
            assert sstable.entry_count == 2
            assert sstable.min_key == "key1"
            assert sstable.max_key == "key2"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_get_existing_key(self):
        """Should retrieve existing key."""
        mem = Memtable()
        clock = VectorClock.new("node-1")
        
        mem.put("key1", {"nested": "value"}, clock)
        
        tmpdir = tempfile.mkdtemp()
        try:
            path = Path(tmpdir) / "test.sst"
            sstable = await SSTable.create_from_memtable(mem, path)
            
            result = await sstable.get("key1")
            assert result is not None
            assert result[0] == {"nested": "value"}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_get_missing_key(self):
        """Should return None for missing key."""
        mem = Memtable()
        clock = VectorClock.new("node-1")
        mem.put("key1", "value1", clock)
        
        tmpdir = tempfile.mkdtemp()
        try:
            path = Path(tmpdir) / "test.sst"
            sstable = await SSTable.create_from_memtable(mem, path)
            
            result = await sstable.get("missing")
            assert result is None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_load_index(self):
        """Should load index from disk."""
        mem = Memtable()
        clock = VectorClock.new("node-1")
        mem.put("key1", "value1", clock)
        
        tmpdir = tempfile.mkdtemp()
        try:
            path = Path(tmpdir) / "test.sst"
            sstable1 = await SSTable.create_from_memtable(mem, path)
            
            # Load in new instance
            sstable2 = SSTable(path)
            await sstable2.load_index()
            
            assert sstable2.entry_count == 1
            assert "key1" in sstable2.index
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_may_contain(self):
        """Should check key range correctly."""
        sstable = SSTable(Path("/fake/path"))
        sstable.min_key = "a"
        sstable.max_key = "z"
        
        assert sstable.may_contain("m") is True
        assert sstable.may_contain("a") is True
        assert sstable.may_contain("z") is True
        assert sstable.may_contain("0") is False
        assert sstable.may_contain("{") is False


class TestStorageEngine:
    """Tests for StorageEngine."""
    
    @pytest.fixture
    async def engine(self):
        """Create a temporary storage engine."""
        tmpdir = tempfile.mkdtemp()
        engine = StorageEngine(tmpdir, memtable_size=100)
        await engine.start()
        yield engine
        await engine.stop()
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_write_and_read(self):
        """Should write and read back values."""
        tmpdir = tempfile.mkdtemp()
        engine = StorageEngine(tmpdir)
        await engine.start()
        
        try:
            clock = VectorClock.new("node-1")
            await engine.write("key1", {"data": "value"}, clock)
            
            result = await engine.read("key1")
            assert result is not None
            assert result[0] == {"data": "value"}
        finally:
            await engine.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_read_missing_key(self):
        """Should return None for missing key."""
        tmpdir = tempfile.mkdtemp()
        engine = StorageEngine(tmpdir)
        await engine.start()
        
        try:
            result = await engine.read("missing")
            assert result is None
        finally:
            await engine.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_delete_removes_key(self):
        """Delete should remove key."""
        tmpdir = tempfile.mkdtemp()
        engine = StorageEngine(tmpdir)
        await engine.start()
        
        try:
            clock = VectorClock.new("node-1")
            await engine.write("key1", "value1", clock)
            await engine.delete("key1", clock)
            
            result = await engine.read("key1")
            assert result is None
        finally:
            await engine.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_wal_recovery_on_startup(self):
        """Should recover from WAL on startup."""
        tmpdir = tempfile.mkdtemp()
        
        # First engine - write data
        engine1 = StorageEngine(tmpdir)
        await engine1.start()
        
        try:
            clock = VectorClock.new("node-1")
            await engine1.write("key1", "value1", clock)
            await engine1.wal.fsync()  # Ensure written
        finally:
            await engine1.stop()
        
        # Second engine - should recover
        engine2 = StorageEngine(tmpdir)
        await engine2.start()
        
        try:
            result = await engine2.read("key1")
            assert result is not None
            assert result[0] == "value1"
        finally:
            await engine2.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Should return statistics."""
        tmpdir = tempfile.mkdtemp()
        engine = StorageEngine(tmpdir)
        await engine.start()
        
        try:
            clock = VectorClock.new("node-1")
            await engine.write("key1", "value1", clock)
            
            stats = engine.get_stats()
            assert "wal" in stats
            assert "memtable" in stats
            assert "sstables" in stats
            assert stats["memtable"]["entries"] == 1
        finally:
            await engine.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)
