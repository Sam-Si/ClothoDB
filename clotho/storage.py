"""
Storage layer for ClothoDB: WAL, Memtable, and SSTable implementation.

This module implements the hybrid write path:
1. Write-Ahead Log (WAL) for durability
2. Memtable for fast in-memory reads
3. SSTable for persistent storage
4. Group commit for efficient fsync
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import aiofiles

from .vector_clock import VectorClock


# =============================================================================
# Write-Ahead Log (WAL)
# =============================================================================

@dataclass
class WALEntry:
    """Single entry in the write-ahead log."""
    sequence_number: int
    timestamp: float
    key: str
    value: Any
    vector_clock: Dict[str, int]
    operation: str = "WRITE"  # WRITE or DELETE
    
    def serialize(self) -> bytes:
        """Serialize entry to bytes for writing to WAL."""
        data = {
            "seq": self.sequence_number,
            "ts": self.timestamp,
            "key": self.key,
            "value": self.value,
            "vc": self.vector_clock,
            "op": self.operation
        }
        json_bytes = json.dumps(data, separators=(',', ':')).encode('utf-8')
        # Format: [4 bytes length][json data][4 bytes checksum]
        length = len(json_bytes)
        checksum = self._checksum(json_bytes)
        return struct.pack('>I', length) + json_bytes + struct.pack('>I', checksum)
    
    @classmethod
    def deserialize(cls, data: bytes) -> WALEntry:
        """Deserialize entry from bytes."""
        if len(data) < 8:
            raise ValueError("Data too short for WAL entry")
        
        length = struct.unpack('>I', data[:4])[0]
        json_bytes = data[4:4+length]
        stored_checksum = struct.unpack('>I', data[4+length:8+length])[0]
        
        # Verify checksum
        if cls._checksum(json_bytes) != stored_checksum:
            raise ValueError("Checksum mismatch in WAL entry")
        
        payload = json.loads(json_bytes.decode('utf-8'))
        return cls(
            sequence_number=payload["seq"],
            timestamp=payload["ts"],
            key=payload["key"],
            value=payload["value"],
            vector_clock=payload["vc"],
            operation=payload.get("op", "WRITE")
        )
    
    @staticmethod
    def _checksum(data: bytes) -> int:
        """Simple checksum for corruption detection."""
        return sum(data) % (2**32)


class WriteAheadLog:
    """
    Write-Ahead Log for durability.
    
    All writes are appended to the WAL before being applied to the memtable.
    The WAL is fsync'd periodically (group commit) to amortize disk I/O cost.
    """
    
    def __init__(
        self,
        log_dir: str,
        max_file_size: int = 64 * 1024 * 1024,  # 64MB
        group_commit_interval_ms: float = 10.0
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        self.group_commit_interval_ms = group_commit_interval_ms
        
        self.current_file: Optional[aiofiles.threadpool.AsyncBufferedIOBase] = None
        self.current_file_path: Optional[Path] = None
        self.current_file_size = 0
        self.sequence_number = 0
        
        # Group commit state
        self._pending_fsync = False
        self._fsync_event = asyncio.Event()
        self._fsync_event.set()  # Initially ready
        self._group_commit_task: Optional[asyncio.Task] = None
        
        # Stats
        self.entries_written = 0
        self.bytes_written = 0
    
    async def start(self):
        """Start the WAL and group commit background task."""
        await self._rotate_file()
        self._group_commit_task = asyncio.create_task(self._group_commit_loop())
    
    async def stop(self):
        """Stop the WAL and ensure all data is flushed."""
        if self._group_commit_task:
            self._group_commit_task.cancel()
            try:
                await self._group_commit_task
            except asyncio.CancelledError:
                pass
        
        if self.current_file:
            await self.current_file.flush()
            await self.current_file.close()
    
    async def append(
        self,
        key: str,
        value: Any,
        vector_clock: VectorClock,
        operation: str = "WRITE"
    ) -> int:
        """
        Append an entry to the WAL.
        
        Returns the sequence number of the entry.
        The entry may not be fsync'd yet (group commit).
        """
        self.sequence_number += 1
        
        entry = WALEntry(
            sequence_number=self.sequence_number,
            timestamp=time.time(),
            key=key,
            value=value,
            vector_clock=vector_clock.to_dict(),
            operation=operation
        )
        
        data = entry.serialize()
        
        # Check if we need to rotate
        if self.current_file_size + len(data) > self.max_file_size:
            await self._rotate_file()
        
        # Write to current file
        await self.current_file.write(data)
        self.current_file_size += len(data)
        
        # Mark that we need fsync
        self._pending_fsync = True
        self._fsync_event.clear()
        
        self.entries_written += 1
        self.bytes_written += len(data)
        
        return self.sequence_number
    
    async def fsync(self):
        """Force fsync of the current WAL file."""
        if self.current_file and self._pending_fsync:
            try:
                await self.current_file.flush()
            except ValueError:
                # File was closed (e.g., during rotation)
                pass
            self._pending_fsync = False
            self._fsync_event.set()
    
    async def wait_for_fsync(self):
        """Wait for the next group commit fsync."""
        await self._fsync_event.wait()
    
    async def _rotate_file(self):
        """Rotate to a new WAL file."""
        if self.current_file:
            await self.current_file.flush()
            # Mark that fsync is no longer pending
            self._pending_fsync = False
            self._fsync_event.set()
            await self.current_file.close()
        
        timestamp = int(time.time() * 1000)
        filename = f"wal-{timestamp:020d}.log"
        self.current_file_path = self.log_dir / filename
        
        self.current_file = await aiofiles.open(
            self.current_file_path,
            mode='wb'
        )
        self.current_file_size = 0
    
    async def _group_commit_loop(self):
        """Background task to periodically fsync the WAL."""
        while True:
            try:
                await asyncio.sleep(self.group_commit_interval_ms / 1000.0)
                if self._pending_fsync:
                    await self.fsync()
            except asyncio.CancelledError:
                # Final fsync before exit
                if self._pending_fsync:
                    await self.fsync()
                break
    
    async def recover(self) -> List[WALEntry]:
        """
        Recover entries from WAL files.
        
        Called on startup to replay unflushed writes.
        """
        entries = []
        
        # Find all WAL files
        wal_files = sorted(self.log_dir.glob("wal-*.log"))
        
        for wal_file in wal_files:
            async with aiofiles.open(wal_file, mode='rb') as f:
                data = await f.read()
            
            offset = 0
            while offset < len(data):
                try:
                    # Read length
                    if offset + 4 > len(data):
                        break
                    length = struct.unpack('>I', data[offset:offset+4])[0]
                    
                    # Read full entry
                    entry_end = offset + 8 + length
                    if entry_end > len(data):
                        break
                    
                    entry_data = data[offset:entry_end]
                    entry = WALEntry.deserialize(entry_data)
                    entries.append(entry)
                    
                    offset = entry_end
                except Exception as e:
                    # Corrupted entry, stop here
                    print(f"WAL recovery error at offset {offset}: {e}")
                    break
        
        return entries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WAL statistics."""
        return {
            "entries_written": self.entries_written,
            "bytes_written": self.bytes_written,
            "current_file": str(self.current_file_path) if self.current_file_path else None,
            "current_file_size": self.current_file_size,
            "pending_fsync": self._pending_fsync
        }


# =============================================================================
# Memtable
# =============================================================================

class Memtable:
    """
    In-memory sorted map for fast writes and reads.
    
    Uses OrderedDict to maintain insertion order (simplified skiplist).
    When full, flushed to SSTable.
    """
    
    def __init__(self, max_size: int = 10000):
        self.data: OrderedDict[str, Tuple[Any, VectorClock]] = OrderedDict()
        self.max_size = max_size
        self.write_count = 0
    
    def put(self, key: str, value: Any, vector_clock: VectorClock):
        """Insert or update a key."""
        self.data[key] = (value, vector_clock)
        self.write_count += 1
        
        # Move to end (most recent)
        self.data.move_to_end(key)
    
    def get(self, key: str) -> Optional[Tuple[Any, VectorClock]]:
        """Get value and vector clock for a key."""
        return self.data.get(key)
    
    def delete(self, key: str):
        """Delete a key (tombstone)."""
        if key in self.data:
            del self.data[key]
            self.write_count += 1
    
    def is_full(self) -> bool:
        """Check if memtable should be flushed."""
        return len(self.data) >= self.max_size
    
    def items(self):
        """Iterate over all items in sorted order."""
        return sorted(self.data.items())
    
    def clear(self):
        """Clear all data."""
        self.data.clear()
        self.write_count = 0
    
    def __len__(self) -> int:
        return len(self.data)


# =============================================================================
# SSTable
# =============================================================================

@dataclass
class SSTableIndexEntry:
    """Index entry for fast key lookup in SSTable."""
    key: str
    offset: int
    length: int


class SSTable:
    """
    Sorted String Table - immutable on-disk storage.
    
    Format:
    - Data section: sequence of [key_len][key][value_len][value][clock_len][clock]
    - Index section: sorted list of key->offset mappings
    - Footer: offset of index start
    """
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.index: Dict[str, Tuple[int, int]] = {}  # key -> (offset, length)
        self.min_key: Optional[str] = None
        self.max_key: Optional[str] = None
        self.entry_count = 0
    
    @classmethod
    async def create_from_memtable(
        cls,
        memtable: Memtable,
        file_path: Path
    ) -> SSTable:
        """Flush a memtable to disk as SSTable."""
        sstable = cls(file_path)
        
        # Build data and index
        data_parts = []
        index = {}
        current_offset = 0
        
        for key, (value, clock) in memtable.items():
            # Serialize value and clock
            value_json = json.dumps(value).encode('utf-8')
            clock_json = json.dumps(clock.to_dict()).encode('utf-8')
            
            # Build entry: [key_len][key][value_len][value][clock_len][clock]
            key_bytes = key.encode('utf-8')
            entry = struct.pack('>I', len(key_bytes)) + key_bytes
            entry += struct.pack('>I', len(value_json)) + value_json
            entry += struct.pack('>I', len(clock_json)) + clock_json
            
            index[key] = (current_offset, len(entry))
            data_parts.append(entry)
            current_offset += len(entry)
        
        # Write to file
        async with aiofiles.open(file_path, mode='wb') as f:
            # Write data section
            for part in data_parts:
                await f.write(part)
            
            # Write index
            index_offset = current_offset
            index_data = json.dumps(index).encode('utf-8')
            await f.write(struct.pack('>I', len(index_data)))
            await f.write(index_data)
            
            # Write footer (index offset)
            await f.write(struct.pack('>Q', index_offset))
        
        # Update sstable metadata
        sstable.index = index
        sstable.entry_count = len(index)
        if index:
            sstable.min_key = min(index.keys())
            sstable.max_key = max(index.keys())
        
        return sstable
    
    async def load_index(self):
        """Load the index from disk."""
        async with aiofiles.open(self.file_path, mode='rb') as f:
            # Seek to footer (last 8 bytes)
            await f.seek(-8, 2)
            footer = await f.read(8)
            index_offset = struct.unpack('>Q', footer)[0]
            
            # Read index
            await f.seek(index_offset)
            index_len_data = await f.read(4)
            index_len = struct.unpack('>I', index_len_data)[0]
            index_data = await f.read(index_len)
            
            self.index = json.loads(index_data.decode('utf-8'))
            self.entry_count = len(self.index)
            if self.index:
                self.min_key = min(self.index.keys())
                self.max_key = max(self.index.keys())
    
    async def get(self, key: str) -> Optional[Tuple[Any, VectorClock]]:
        """Get a value from the SSTable."""
        if key not in self.index:
            return None
        
        offset, length = self.index[key]
        
        async with aiofiles.open(self.file_path, mode='rb') as f:
            await f.seek(offset)
            data = await f.read(length)
        
        # Parse entry
        pos = 0
        
        # Read key
        key_len = struct.unpack('>I', data[pos:pos+4])[0]
        pos += 4 + key_len
        
        # Read value
        value_len = struct.unpack('>I', data[pos:pos+4])[0]
        pos += 4
        value = json.loads(data[pos:pos+value_len].decode('utf-8'))
        pos += value_len
        
        # Read clock
        clock_len = struct.unpack('>I', data[pos:pos+4])[0]
        pos += 4
        clock_dict = json.loads(data[pos:pos+clock_len].decode('utf-8'))
        clock = VectorClock.from_dict(clock_dict)
        
        return value, clock
    
    def may_contain(self, key: str) -> bool:
        """Check if key might be in this SSTable (using key range)."""
        if self.min_key is None:
            return False
        return self.min_key <= key <= self.max_key
    
    async def delete(self):
        """Delete the SSTable file."""
        if self.file_path.exists():
            os.remove(self.file_path)


# =============================================================================
# Storage Engine
# =============================================================================

class StorageEngine:
    """
    Combined storage engine: WAL + Memtable + SSTables.
    
    Write path:
    1. Append to WAL
    2. Update memtable
    3. (Async) Flush memtable to SSTable when full
    
    Read path:
    1. Check memtable
    2. Check SSTables (newest first)
    """
    
    def __init__(
        self,
        data_dir: str,
        memtable_size: int = 10000,
        wal_group_commit_ms: float = 10.0
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.wal = WriteAheadLog(
            str(self.data_dir / "wal"),
            group_commit_interval_ms=wal_group_commit_ms
        )
        self.memtable = Memtable(max_size=memtable_size)
        self.sstables: List[SSTable] = []
        
        self._flush_lock = asyncio.Lock()
        self._running = False
    
    async def start(self):
        """Start the storage engine and recover from WAL."""
        self._running = True
        
        # Start WAL
        await self.wal.start()
        
        # Recover from WAL
        entries = await self.wal.recover()
        for entry in entries:
            if entry.operation == "WRITE":
                clock = VectorClock.from_dict(entry.vector_clock)
                self.memtable.put(entry.key, entry.value, clock)
            elif entry.operation == "DELETE":
                self.memtable.delete(entry.key)
        
        print(f"StorageEngine recovered {len(entries)} entries from WAL")
        
        # Load existing SSTables
        await self._load_sstables()
    
    async def stop(self):
        """Stop the storage engine."""
        self._running = False
        
        # Flush memtable if not empty
        if len(self.memtable) > 0:
            await self._flush_memtable()
        
        await self.wal.stop()
    
    async def write(
        self,
        key: str,
        value: Any,
        vector_clock: VectorClock
    ) -> bool:
        """
        Write a key-value pair.
        
        1. Append to WAL
        2. Update memtable
        3. Check if memtable needs flush
        """
        # 1. Append to WAL
        await self.wal.append(key, value, vector_clock)
        
        # 2. Update memtable
        self.memtable.put(key, value, vector_clock)
        
        # 3. Check if memtable is full
        if self.memtable.is_full():
            asyncio.create_task(self._flush_memtable())
        
        return True
    
    async def read(self, key: str) -> Optional[Tuple[Any, VectorClock]]:
        """
        Read a key-value pair.
        
        1. Check memtable (most recent)
        2. Check SSTables (newest first)
        """
        # 1. Check memtable
        result = self.memtable.get(key)
        if result:
            return result
        
        # 2. Check SSTables (newest first)
        for sstable in reversed(self.sstables):
            if sstable.may_contain(key):
                result = await sstable.get(key)
                if result:
                    return result
        
        return None
    
    async def delete(self, key: str, vector_clock: VectorClock):
        """Delete a key (tombstone)."""
        await self.wal.append(key, None, vector_clock, operation="DELETE")
        self.memtable.delete(key)
    
    async def _flush_memtable(self):
        """Flush memtable to SSTable."""
        async with self._flush_lock:
            if len(self.memtable) == 0:
                return
            
            # Create new SSTable
            timestamp = int(time.time() * 1000)
            sstable_path = self.data_dir / f"sstable-{timestamp:020d}.db"
            
            sstable = await SSTable.create_from_memtable(
                self.memtable,
                sstable_path
            )
            
            self.sstables.append(sstable)
            
            # Clear memtable
            self.memtable.clear()
            
            print(f"Flushed memtable to {sstable_path} ({sstable.entry_count} entries)")
    
    async def _load_sstables(self):
        """Load existing SSTables from disk."""
        sstable_files = sorted(self.data_dir.glob("sstable-*.db"))
        
        for file_path in sstable_files:
            sstable = SSTable(file_path)
            await sstable.load_index()
            self.sstables.append(sstable)
        
        print(f"Loaded {len(self.sstables)} SSTables")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage engine statistics."""
        return {
            "wal": self.wal.get_stats(),
            "memtable": {
                "entries": len(self.memtable),
                "max_size": self.memtable.max_size
            },
            "sstables": {
                "count": len(self.sstables),
                "total_entries": sum(s.entry_count for s in self.sstables)
            }
        }
