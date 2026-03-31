#!/usr/bin/env python3
"""
Demo: ClothoDB Write Path

Demonstrates the hybrid write path implementation:
- Write-Ahead Log (WAL) with group commit
- Quorum-based writes (N=3, W=2)
- Memtable + SSTable storage
- Vector clock tracking
"""

import asyncio
import tempfile
import shutil
from clotho import create_5_server_cluster, DistributedWriteClient


async def demo_basic_write():
    """Demo: Basic write operation."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Write Operation")
    print("=" * 70)
    
    # Create temp directory for data
    data_dir = tempfile.mkdtemp()
    
    try:
        # Create 5-server cluster
        router = create_5_server_cluster(base_port=8000)
        
        # Create write client
        client = DistributedWriteClient(
            router=router,
            data_dir=data_dir,
            n=3,
            w=2
        )
        
        # Start client
        await client.start()
        
        # Write some data
        print("\nWriting 5 key-value pairs...")
        for i in range(5):
            key = f"user:{i}"
            value = {"name": f"User {i}", "index": i}
            
            result = await client.put(key, value)
            
            if result.success:
                print(f"  ✓ {key}: Written to {result.replicas_written} replicas "
                      f"(latency: {result.latency_ms:.2f}ms)")
            else:
                print(f"  ✗ {key}: Failed - {result.error}")
        
        # Show stats
        print("\nWrite Statistics:")
        stats = client.get_stats()
        coord_stats = stats["coordinator"]
        print(f"  Total writes: {coord_stats['writes_total']}")
        print(f"  Successful: {coord_stats['writes_successful']}")
        print(f"  Failed: {coord_stats['writes_failed']}")
        print(f"  Success rate: {coord_stats['success_rate']:.1f}%")
        
        # Show storage stats
        print("\nStorage Statistics:")
        for node_id, node_stats in stats["storage"].items():
            print(f"  {node_id}:")
            print(f"    WAL entries: {node_stats['wal']['entries_written']}")
            print(f"    Memtable: {node_stats['memtable']['entries']} entries")
            print(f"    SSTables: {node_stats['sstables']['count']}")
        
        # Read back the data
        print("\nReading back data...")
        for i in range(5):
            key = f"user:{i}"
            result = await client.get(key)
            if result:
                value, clock = result
                print(f"  ✓ {key}: {value}")
            else:
                print(f"  ✗ {key}: Not found")
        
        await client.stop()
        
    finally:
        # Cleanup
        shutil.rmtree(data_dir, ignore_errors=True)


async def demo_quorum_failures():
    """Demo: Quorum behavior during failures."""
    print("\n" + "=" * 70)
    print("DEMO 2: Quorum Behavior (N=3, W=2)")
    print("=" * 70)
    
    data_dir = tempfile.mkdtemp()
    
    try:
        router = create_5_server_cluster(base_port=8000)
        client = DistributedWriteClient(router, data_dir, n=3, w=2)
        await client.start()
        
        print("\nWith N=3 replicas and W=2 quorum:")
        print("  - Write succeeds if 2+ replicas acknowledge")
        print("  - Write fails if fewer than 2 replicas respond")
        print("  - Hints stored for failed replicas")
        
        # Write with normal conditions
        print("\nWriting under normal conditions...")
        result = await client.put("test:1", {"data": "value1"})
        print(f"  Result: {result.success}, Replicas: {result.replicas_written}")
        
        # Show hint stats
        stats = client.get_stats()
        hints = stats["coordinator"]["hinted_handoff"]
        print(f"  Pending hints: {hints['total_pending_hints']}")
        
        await client.stop()
        
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


async def demo_wal_recovery():
    """Demo: WAL recovery after restart."""
    print("\n" + "=" * 70)
    print("DEMO 3: WAL Recovery")
    print("=" * 70)
    
    data_dir = tempfile.mkdtemp()
    
    try:
        # Phase 1: Write data
        print("\nPhase 1: Writing data...")
        router = create_5_server_cluster(base_port=8000)
        client = DistributedWriteClient(router, data_dir, n=3, w=2)
        await client.start()
        
        for i in range(3):
            await client.put(f"recover:{i}", {"value": i})
        
        # Get stats before stop
        stats_before = client.get_stats()
        total_wal_before = sum(
            s["wal"]["entries_written"]
            for s in stats_before["storage"].values()
        )
        print(f"  Written {total_wal_before} WAL entries")
        
        await client.stop()
        
        # Phase 2: Restart and recover
        print("\nPhase 2: Restarting and recovering...")
        router2 = create_5_server_cluster(base_port=8000)
        client2 = DistributedWriteClient(router2, data_dir, n=3, w=2)
        await client2.start()
        
        # Read recovered data
        print("\nReading recovered data...")
        for i in range(3):
            key = f"recover:{i}"
            result = await client2.get(key)
            if result:
                value, _ = result
                print(f"  ✓ {key}: {value}")
            else:
                print(f"  ✗ {key}: Not found")
        
        await client2.stop()
        
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


async def demo_vector_clocks():
    """Demo: Vector clock tracking."""
    print("\n" + "=" * 70)
    print("DEMO 4: Vector Clock Tracking")
    print("=" * 70)
    
    data_dir = tempfile.mkdtemp()
    
    try:
        router = create_5_server_cluster(base_port=8000)
        client = DistributedWriteClient(router, data_dir, n=3, w=2)
        await client.start()
        
        print("\nWriting with vector clock tracking...")
        
        # First write
        result1 = await client.put("vc:test", {"version": 1})
        print(f"  Write 1: {result1.vector_clock}")
        
        # Second write (should increment)
        result2 = await client.put("vc:test", {"version": 2})
        print(f"  Write 2: {result2.vector_clock}")
        
        # Show causality
        from clotho import VectorClock
        clock1 = VectorClock.from_dict(result1.vector_clock)
        clock2 = VectorClock.from_dict(result2.vector_clock)
        relation = clock1.compare(clock2)
        print(f"\n  Causality: Write 1 {relation.name} Write 2")
        
        await client.stop()
        
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


async def main():
    """Run all demos."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           ClothoDB: Write Path Implementation Demo                   ║
║                                                                      ║
║  Features: WAL + Group Commit | Quorum Writes | Vector Clocks       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    await demo_basic_write()
    await demo_quorum_failures()
    await demo_wal_recovery()
    await demo_vector_clocks()
    
    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
