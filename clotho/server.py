"""
HTTP Server for ClothoDB nodes.

Each server runs on a specific port and:
1. Stores data for keys assigned to it via consistent hashing
2. Participates in the vector clock protocol for causality tracking
3. Provides REST API for get/put/delete operations
4. Provides routing API to find which nodes own a key
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional, List, Any, Set
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from .node import Node, Event
from .vector_clock import VectorClock
from .consistent_hash import Router, NodeInfo, create_5_server_cluster


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class PutRequest(BaseModel):
    key: str
    value: Any
    context: Optional[Dict[str, int]] = None  # Vector clock context from client


class GetResponse(BaseModel):
    key: str
    value: Optional[Any]
    node_id: str
    vector_clock: Dict[str, int]
    found: bool


class PutResponse(BaseModel):
    key: str
    value: Any
    node_id: str
    vector_clock: Dict[str, int]
    success: bool


class DeleteResponse(BaseModel):
    key: str
    node_id: str
    vector_clock: Dict[str, int]
    deleted: bool


class RoutingResponse(BaseModel):
    key: str
    primary: Optional[Dict[str, Any]]
    replicas: List[Dict[str, Any]]
    replication_factor: int
    success: bool


class NodeStatus(BaseModel):
    node_id: str
    address: str
    event_count: int
    data_keys: int
    known_nodes: List[str]
    vector_clock: Dict[str, int]


class ClusterStatus(BaseModel):
    servers: List[Dict[str, Any]]
    total_keys: int
    replication_factor: int


class ClothoServer:
    """
    HTTP Server wrapping a ClothoDB node.
    
    Each server:
    - Runs on a specific host:port
    - Has a unique node_id
    - Stores key-value pairs with vector clock metadata
    - Participates in consistent hashing for routing
    """
    
    def __init__(
        self,
        node_id: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        router: Optional[Router] = None
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.node = Node(node_id)
        self.router = router
        
        # Track which keys this node is responsible for
        self._owned_keys: Set[str] = set()
        
        # FastAPI app
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info(f"Server {self.node_id} starting on {self.host}:{self.port}")
            yield
            logger.info(f"Server {self.node_id} shutting down")
        
        app = FastAPI(
            title=f"ClothoDB - {self.node_id}",
            description=f"Distributed node {self.node_id} at {self.host}:{self.port}",
            version="0.1.0",
            lifespan=lifespan
        )
        
        # ========== Data Operations ==========
        
        @app.get("/get/{key}", response_model=GetResponse)
        async def get_key(key: str) -> GetResponse:
            """
            Get a value by key.
            
            Returns the value along with its vector clock for causality tracking.
            """
            result = self.node.read(key)
            
            if result is None:
                return GetResponse(
                    key=key,
                    value=None,
                    node_id=self.node_id,
                    vector_clock=self.node.vector_clock.to_dict(),
                    found=False
                )
            
            value, vc = result
            return GetResponse(
                key=key,
                value=value,
                node_id=self.node_id,
                vector_clock=vc.to_dict(),
                found=True
            )
        
        @app.post("/put", response_model=PutResponse)
        async def put_key(request: PutRequest) -> PutResponse:
            """
            Store a key-value pair.
            
            If context (vector clock) is provided, merges it before writing
            to maintain causality.
            """
            # If client provides context, merge it first
            if request.context:
                client_clock = VectorClock.from_dict(request.context)
                self.node.receive_message("client", client_clock)
            
            # Perform the write
            event = self.node.write(request.key, request.value)
            self._owned_keys.add(request.key)
            
            return PutResponse(
                key=request.key,
                value=request.value,
                node_id=self.node_id,
                vector_clock=event.vector_clock.to_dict(),
                success=True
            )
        
        @app.delete("/delete/{key}", response_model=DeleteResponse)
        async def delete_key(key: str) -> DeleteResponse:
            """Delete a key."""
            existed = key in self.node.data
            event = self.node.delete(key)
            self._owned_keys.discard(key)
            
            return DeleteResponse(
                key=key,
                node_id=self.node_id,
                vector_clock=event.vector_clock.to_dict(),
                deleted=existed
            )
        
        # ========== Routing Operations ==========
        
        @app.get("/route/{key}", response_model=RoutingResponse)
        async def route_key(key: str) -> RoutingResponse:
            """
            Get routing information for a key.
            
            Returns the primary node and replica nodes responsible for the key.
            """
            if self.router is None:
                raise HTTPException(
                    status_code=503,
                    detail="Router not configured"
                )
            
            routing_info = self.router.route_key(key)
            
            # Convert NodeInfo to dict for JSON serialization
            def node_to_dict(n):
                return {
                    "node_id": n.node_id,
                    "host": n.host,
                    "port": n.port,
                    "address": n.address
                } if n else None
            
            return RoutingResponse(
                key=key,
                primary=node_to_dict(routing_info["primary"]),
                replicas=[node_to_dict(r) for r in routing_info["replicas"]],
                replication_factor=routing_info["replication_factor"],
                success=routing_info["success"]
            )
        
        @app.get("/top3/{key}")
        async def get_top3_nodes(key: str) -> Dict[str, Any]:
            """
            Get the top 3 nodes responsible for a key.
            
            This is a convenience endpoint for the specific requirement.
            """
            if self.router is None:
                raise HTTPException(
                    status_code=503,
                    detail="Router not configured"
                )
            
            nodes = self.router.get_replica_nodes(key)
            
            return {
                "key": key,
                "top_3_nodes": [
                    {
                        "rank": i + 1,
                        "node_id": n.node_id,
                        "host": n.host,
                        "port": n.port,
                        "address": n.address,
                        "is_primary": i == 0
                    }
                    for i, n in enumerate(nodes[:3])
                ],
                "total_found": len(nodes)
            }
        
        # ========== Status & Info ==========
        
        @app.get("/status", response_model=NodeStatus)
        async def get_status() -> NodeStatus:
            """Get the current status of this node."""
            return NodeStatus(
                node_id=self.node_id,
                address=f"{self.host}:{self.port}",
                event_count=len(self.node.events),
                data_keys=len(self.node.data),
                known_nodes=list(self.node.known_nodes),
                vector_clock=self.node.vector_clock.to_dict()
            )
        
        @app.get("/keys")
        async def get_keys() -> Dict[str, Any]:
            """Get all keys stored on this node."""
            return {
                "node_id": self.node_id,
                "keys": list(self.node.data.keys()),
                "count": len(self.node.data)
            }
        
        @app.get("/events")
        async def get_events() -> Dict[str, Any]:
            """Get event history for this node."""
            events = [
                {
                    "event_id": e.event_id,
                    "operation": e.operation,
                    "key": e.key,
                    "value": e.value,
                    "vector_clock": e.vector_clock.to_dict()
                }
                for e in self.node.events
            ]
            
            return {
                "node_id": self.node_id,
                "events": events,
                "count": len(events)
            }
        
        @app.post("/gossip/{target_node_id}")
        async def gossip(target_node_id: str, background_tasks: BackgroundTasks):
            """
            Trigger a gossip/anti-entropy session with another node.
            
            In a real implementation, this would sync data.
            Here we just update vector clocks.
            """
            # This is a placeholder for actual gossip implementation
            return {
                "status": "gossip_initiated",
                "from": self.node_id,
                "to": target_node_id
            }
        
        # ========== Cluster Operations ==========
        
        @app.get("/cluster/status")
        async def get_cluster_status() -> ClusterStatus:
            """Get status of the entire cluster."""
            if self.router is None:
                raise HTTPException(
                    status_code=503,
                    detail="Router not configured"
                )
            
            servers = self.router.get_all_servers()
            
            return ClusterStatus(
                servers=[
                    {
                        "node_id": s.node_id,
                        "host": s.host,
                        "port": s.port,
                        "address": s.address
                    }
                    for s in servers
                ],
                total_keys=sum(len(self.node.data) for _ in servers),
                replication_factor=self.router.replication_factor
            )
        
        @app.get("/cluster/distribution")
        async def get_cluster_distribution() -> Dict[str, Any]:
            """Get distribution statistics for the cluster."""
            if self.router is None:
                raise HTTPException(
                    status_code=503,
                    detail="Router not configured"
                )
            
            return self.router.get_distribution_stats()
        
        return app
    
    def run(self, reload: bool = False):
        """Run the server (blocking)."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=reload,
            log_level="info"
        )
    
    async def start(self):
        """Start the server asynchronously (non-blocking)."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    def is_responsible_for(self, key: str) -> bool:
        """Check if this node is responsible for a given key."""
        if self.router is None:
            return True  # If no router, assume responsible
        
        primary = self.router.get_primary_node(key)
        return primary is not None and primary.node_id == self.node_id
    
    def __repr__(self) -> str:
        return f"ClothoServer({self.node_id}@{self.host}:{self.port})"


class Cluster:
    """
    Manages a cluster of ClothoDB servers.
    
    Provides convenient methods for starting/stopping multiple servers
    and routing requests.
    """
    
    def __init__(
        self,
        base_port: int = 8000,
        num_servers: int = 5,
        replication_factor: int = 3
    ):
        self.base_port = base_port
        self.num_servers = num_servers
        self.replication_factor = replication_factor
        
        # Create the shared router
        self.router = Router(replication_factor=replication_factor)
        
        # Create servers
        self.servers: Dict[str, ClothoServer] = {}
        for i in range(num_servers):
            node_id = f"server-{i+1}"
            port = base_port + i
            
            server = ClothoServer(
                node_id=node_id,
                host="127.0.0.1",
                port=port,
                router=self.router
            )
            self.servers[node_id] = server
            
            # Add to router
            self.router.add_server(node_id, "127.0.0.1", port)
    
    def get_server(self, node_id: str) -> Optional[ClothoServer]:
        """Get a server by node_id."""
        return self.servers.get(node_id)
    
    def get_server_for_key(self, key: str) -> Optional[ClothoServer]:
        """Get the primary server responsible for a key."""
        node_info = self.router.get_primary_node(key)
        if node_info:
            return self.servers.get(node_info.node_id)
        return None
    
    def get_top3_for_key(self, key: str) -> List[ClothoServer]:
        """Get the top 3 servers responsible for a key."""
        nodes = self.router.get_replica_nodes(key)
        return [
            self.servers[n.node_id]
            for n in nodes[:3]
            if n.node_id in self.servers
        ]
    
    def route_key(self, key: str) -> Dict[str, Any]:
        """Get routing information for a key."""
        return self.router.route_key(key)
    
    async def start_all(self):
        """Start all servers concurrently."""
        tasks = [
            asyncio.create_task(server.start())
            for server in self.servers.values()
        ]
        await asyncio.gather(*tasks)
    
    def run_server(self, node_id: str):
        """Run a single server (blocking)."""
        server = self.servers.get(node_id)
        if server:
            server.run()
    
    def print_routing_table(self):
        """Print the routing table for all servers."""
        print("\n" + "=" * 60)
        print("ClothoDB Cluster Routing Table")
        print("=" * 60)
        print(f"\nBase Port: {self.base_port}")
        print(f"Servers: {self.num_servers}")
        print(f"Replication Factor: {self.replication_factor}")
        
        print("\nServers:")
        for node_id, server in self.servers.items():
            print(f"  {node_id}: http://{server.host}:{server.port}")
        
        print("\nVirtual Node Distribution:")
        stats = self.router.get_distribution_stats()
        for node_id, count in stats.get("distribution", {}).items():
            print(f"  {node_id}: {count} vnodes")
        
        print("\n" + "=" * 60)


def get_top3_nodes_for_key(key: str, router: Router) -> List[NodeInfo]:
    """
    Get the top 3 nodes responsible for a given key.
    
    This is the main API function requested.
    
    Args:
        key: The key to route
        router: The Router instance
    
    Returns:
        List of up to 3 NodeInfo objects
    """
    return router.ring.get_nodes(key, n=3)


# Convenience function for the specific requirement
def get_top3_nodes_api(key: str, base_port: int = 8000) -> Dict[str, Any]:
    """
    API function that returns the top 3 nodes for a key.
    
    Creates a 5-server cluster configuration and returns routing info.
    
    Args:
        key: The key to look up
        base_port: Starting port for the cluster (default 8000)
    
    Returns:
        Dictionary with routing information
    """
    router = create_5_server_cluster(base_port=base_port)
    nodes = router.get_replica_nodes(key)
    
    return {
        "key": key,
        "top_3_nodes": [
            {
                "rank": i + 1,
                "node_id": n.node_id,
                "host": n.host,
                "port": n.port,
                "address": n.address,
                "is_primary": i == 0
            }
            for i, n in enumerate(nodes[:3])
        ],
        "total_nodes_in_cluster": router.get_server_count(),
        "replication_factor": router.replication_factor
    }
