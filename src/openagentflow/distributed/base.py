"""Base classes for distributed compute across OpenAgentFlow.

Provides the foundational abstractions for running LLM inference (Ollama and
other providers) across heterogeneous compute backends -- Kubernetes pods,
Docker containers, SSH-accessible bare-metal servers, or plain HTTP endpoints.

Design philosophy (inherited from TwinGraph 1.0):
    - Same agent code runs unchanged on local, Docker, K8s, or SSH targets
    - Compute topology is described declaratively via ComputeCluster
    - Load balancing and failover are transparent to callers
    - Zero external dependencies -- stdlib only (urllib, json, asyncio)

Typical usage::

    from openagentflow.distributed import (
        ComputeBackend,
        ComputeCluster,
        ComputeNode,
        LeastLoadBalancer,
    )

    cluster = ComputeCluster("gpu-pool")
    cluster.add_node(
        ComputeNode("gpu1", ComputeBackend.HTTP, "http://gpu1:11434", model="llama3")
    )
    cluster.add_node(
        ComputeNode("gpu2", ComputeBackend.HTTP, "http://gpu2:11434", model="llama3")
    )

    balancer = LeastLoadBalancer()
    node = balancer.select_node(cluster, model="llama3")
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ComputeBackend(Enum):
    """Supported compute backends for distributed inference.

    Each variant corresponds to a concrete backend class that knows how to
    discover, health-check, and lifecycle-manage Ollama (or other LLM server)
    instances running on that substrate.
    """

    LOCAL = "local"           # Direct local execution (default, no orchestrator)
    DOCKER = "docker"         # Docker container via Docker Engine API
    KUBERNETES = "kubernetes" # Kubernetes pod / job via K8s API
    SSH = "ssh"               # Remote machine reachable via SSH
    HTTP = "http"             # Generic remote HTTP endpoint


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ComputeNode:
    """Represents a single compute endpoint capable of running LLM inference.

    A node is typically one Ollama server process -- it may live inside a
    Docker container, a Kubernetes pod, a remote VM, or simply be ``localhost``.

    Attributes:
        node_id:        Unique identifier for this node.
        backend:        Which compute substrate hosts this node.
        endpoint:       URL or connection string (e.g. ``http://gpu1:11434``).
        model:          Model currently loaded on this node (empty = any/unknown).
        gpu_memory_mb:  Available GPU memory in megabytes (0 = unknown).
        max_concurrent: Maximum number of concurrent inference requests.
        current_load:   Number of currently active requests.
        healthy:        Whether the last health check succeeded.
        metadata:       Arbitrary backend-specific metadata (pod name, container
                        ID, SSH user, etc.).
    """

    node_id: str
    backend: ComputeBackend
    endpoint: str
    model: str = ""
    gpu_memory_mb: int = 0
    max_concurrent: int = 1
    current_load: int = 0
    healthy: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    # Thread-safe load tracking
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def available(self) -> bool:
        """Return ``True`` if the node is healthy and has capacity."""
        return self.healthy and self.current_load < self.max_concurrent

    def increment_load(self) -> None:
        """Thread-safe increment of the current load counter."""
        with self._lock:
            self.current_load += 1

    def decrement_load(self) -> None:
        """Thread-safe decrement of the current load counter."""
        with self._lock:
            self.current_load = max(0, self.current_load - 1)

    def mark_unhealthy(self) -> None:
        """Mark this node as unhealthy (failed health check or request error)."""
        self.healthy = False
        logger.warning("Node %s marked unhealthy", self.node_id)

    def mark_healthy(self) -> None:
        """Mark this node as healthy (passed health check)."""
        self.healthy = True

    @property
    def load_ratio(self) -> float:
        """Current load as a fraction of capacity (0.0 = idle, 1.0 = full)."""
        if self.max_concurrent <= 0:
            return 1.0
        return self.current_load / self.max_concurrent

    def __repr__(self) -> str:
        status = "healthy" if self.healthy else "UNHEALTHY"
        return (
            f"ComputeNode(id={self.node_id!r}, backend={self.backend.value}, "
            f"endpoint={self.endpoint!r}, model={self.model!r}, "
            f"load={self.current_load}/{self.max_concurrent}, {status})"
        )


@dataclass
class ComputeCluster:
    """A collection of compute nodes for distributed LLM inference.

    The cluster maintains a list of :class:`ComputeNode` instances and provides
    convenience methods for adding, removing, filtering, and selecting nodes.

    Example::

        cluster = ComputeCluster("my-cluster")
        cluster.add_node(ComputeNode("n1", ComputeBackend.HTTP, "http://a:11434"))
        cluster.add_node(ComputeNode("n2", ComputeBackend.HTTP, "http://b:11434"))

        available = cluster.get_available_nodes(model="llama3")
        best = cluster.get_best_node(model="llama3")
    """

    name: str
    nodes: list[ComputeNode] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_node(self, node: ComputeNode) -> None:
        """Add a node to the cluster.

        If a node with the same ``node_id`` already exists it is replaced.

        Args:
            node: The compute node to add.
        """
        with self._lock:
            # Replace existing node with same ID
            self.nodes = [n for n in self.nodes if n.node_id != node.node_id]
            self.nodes.append(node)
            logger.info(
                "Added node %s to cluster %s (%s)",
                node.node_id,
                self.name,
                node.endpoint,
            )

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the cluster by ID.

        Args:
            node_id: Identifier of the node to remove.
        """
        with self._lock:
            before = len(self.nodes)
            self.nodes = [n for n in self.nodes if n.node_id != node_id]
            if len(self.nodes) < before:
                logger.info("Removed node %s from cluster %s", node_id, self.name)
            else:
                logger.warning(
                    "Node %s not found in cluster %s", node_id, self.name
                )

    def get_node(self, node_id: str) -> ComputeNode | None:
        """Retrieve a node by its ID, or ``None`` if not found."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_available_nodes(self, model: str | None = None) -> list[ComputeNode]:
        """Return all available (healthy + has capacity) nodes.

        Args:
            model: If provided, only return nodes whose ``model`` field matches
                   or is empty (indicating they can serve any model).

        Returns:
            List of available compute nodes, possibly empty.
        """
        result: list[ComputeNode] = []
        for node in self.nodes:
            if not node.available:
                continue
            if model and node.model and node.model != model:
                continue
            result.append(node)
        return result

    def get_best_node(self, model: str | None = None) -> ComputeNode | None:
        """Pick the node with the lowest load that can serve *model*.

        Prefers nodes that already have *model* loaded (model affinity) and
        breaks ties by choosing the node with the smallest ``load_ratio``.

        Args:
            model: Requested model name (``None`` means any model).

        Returns:
            The best available node, or ``None`` if no nodes are available.
        """
        available = self.get_available_nodes(model)
        if not available:
            return None

        # Prefer model-affinity nodes, then sort by load ratio
        def sort_key(n: ComputeNode) -> tuple[int, float]:
            affinity = 0 if (model and n.model == model) else 1
            return (affinity, n.load_ratio)

        available.sort(key=sort_key)
        return available[0]

    @property
    def total_nodes(self) -> int:
        """Total number of nodes in the cluster."""
        return len(self.nodes)

    @property
    def healthy_nodes(self) -> int:
        """Number of healthy nodes."""
        return sum(1 for n in self.nodes if n.healthy)

    @property
    def total_capacity(self) -> int:
        """Sum of ``max_concurrent`` across all healthy nodes."""
        return sum(n.max_concurrent for n in self.nodes if n.healthy)

    @property
    def current_total_load(self) -> int:
        """Sum of ``current_load`` across all nodes."""
        return sum(n.current_load for n in self.nodes)

    def __repr__(self) -> str:
        return (
            f"ComputeCluster(name={self.name!r}, "
            f"nodes={self.total_nodes}, healthy={self.healthy_nodes}, "
            f"load={self.current_total_load}/{self.total_capacity})"
        )


# ---------------------------------------------------------------------------
# Load balancer interface and implementations
# ---------------------------------------------------------------------------

class LoadBalancer(ABC):
    """Strategy for distributing inference requests across cluster nodes.

    Subclass this to implement custom load-balancing policies.  The framework
    ships with :class:`RoundRobinBalancer`, :class:`LeastLoadBalancer`, and
    :class:`ModelAffinityBalancer`.
    """

    @abstractmethod
    def select_node(
        self, cluster: ComputeCluster, model: str | None = None
    ) -> ComputeNode | None:
        """Select a node from *cluster* to handle a request.

        Args:
            cluster: The compute cluster to select from.
            model: The model requested for this inference call.

        Returns:
            A :class:`ComputeNode` to route the request to, or ``None`` if no
            node is available.
        """
        ...


class RoundRobinBalancer(LoadBalancer):
    """Simple round-robin node selection.

    Cycles through available nodes in order, skipping unhealthy or
    fully-loaded nodes.  State is maintained via an internal counter.
    """

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()

    def select_node(
        self, cluster: ComputeCluster, model: str | None = None
    ) -> ComputeNode | None:
        """Select the next available node in round-robin order."""
        available = cluster.get_available_nodes(model)
        if not available:
            return None

        with self._lock:
            idx = self._counter % len(available)
            self._counter += 1

        return available[idx]


class LeastLoadBalancer(LoadBalancer):
    """Select the node with the fewest active requests.

    When multiple nodes have the same load, the first one encountered wins.
    This is the default balancer used by :class:`DistributedOllamaProvider`.
    """

    def select_node(
        self, cluster: ComputeCluster, model: str | None = None
    ) -> ComputeNode | None:
        """Select the node with the lowest current load."""
        available = cluster.get_available_nodes(model)
        if not available:
            return None

        return min(available, key=lambda n: n.load_ratio)


class ModelAffinityBalancer(LoadBalancer):
    """Prefer nodes that already have the requested model loaded.

    Falls back to least-load selection among nodes that do not have the model
    loaded (they will need to pull or swap the model).
    """

    def select_node(
        self, cluster: ComputeCluster, model: str | None = None
    ) -> ComputeNode | None:
        """Select a node, preferring those with the requested model loaded."""
        available = cluster.get_available_nodes(model)
        if not available:
            return None

        if model:
            # Partition into affinity (model matches) and non-affinity
            affinity = [n for n in available if n.model == model]
            non_affinity = [n for n in available if n.model != model]

            if affinity:
                return min(affinity, key=lambda n: n.load_ratio)
            if non_affinity:
                return min(non_affinity, key=lambda n: n.load_ratio)

        # No model preference -- just pick least loaded
        return min(available, key=lambda n: n.load_ratio)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class ComputeBackendInterface(ABC):
    """Protocol that all concrete backends (Docker, K8s, SSH) implement.

    Each backend knows how to discover running Ollama instances, perform
    health checks, and optionally start/stop instances on its substrate.
    """

    @abstractmethod
    async def discover_nodes(self) -> list[ComputeNode]:
        """Discover running Ollama instances on this backend.

        Returns:
            List of discovered :class:`ComputeNode` instances.
        """
        ...

    @abstractmethod
    async def health_check(self, node: ComputeNode) -> bool:
        """Check whether a node is healthy and can serve requests.

        Args:
            node: The node to check.

        Returns:
            ``True`` if the node is healthy.
        """
        ...
