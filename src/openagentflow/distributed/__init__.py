"""Distributed compute layer for OpenAgentFlow.

Enables running Ollama (and other LLM providers) across Kubernetes, Docker,
SSH-accessible servers, and generic HTTP endpoints -- inspired by TwinGraph
1.0's multi-compute dispatch pattern.

All backends use only Python stdlib for network communication (``urllib``,
``http.client``, ``subprocess``).  No external dependencies are required.

Quick start::

    from openagentflow.distributed import (
        ComputeBackend,
        ComputeCluster,
        ComputeNode,
        DistributedOllamaProvider,
        LeastLoadBalancer,
    )

    # Build a cluster from known endpoints
    cluster = ComputeCluster("my-cluster")
    cluster.add_node(
        ComputeNode("gpu1", ComputeBackend.HTTP, "http://gpu1:11434", model="llama3")
    )
    cluster.add_node(
        ComputeNode("gpu2", ComputeBackend.HTTP, "http://gpu2:11434", model="llama3")
    )

    # Create a distributed provider with automatic failover
    provider = DistributedOllamaProvider(cluster=cluster)
    result = await provider.generate(messages=msgs, config=config)

    # Or auto-discover from Kubernetes / Docker / SSH
    from openagentflow.distributed import KubernetesBackend, DockerBackend, SSHBackend

    k8s = KubernetesBackend(namespace="ml-inference")
    await provider.auto_discover(k8s)

Backends:
    - :class:`KubernetesBackend` -- discover and scale Ollama pods via K8s API
    - :class:`DockerBackend` -- manage Ollama containers via Docker Engine API
    - :class:`SSHBackend` -- manage Ollama on remote machines via SSH

Load balancers:
    - :class:`RoundRobinBalancer` -- simple round-robin
    - :class:`LeastLoadBalancer` -- fewest active requests (default)
    - :class:`ModelAffinityBalancer` -- prefer nodes with the model loaded
"""

from openagentflow.distributed.base import (
    ComputeBackend,
    ComputeBackendInterface,
    ComputeCluster,
    ComputeNode,
    LeastLoadBalancer,
    LoadBalancer,
    ModelAffinityBalancer,
    RoundRobinBalancer,
)
from openagentflow.distributed.docker import DockerBackend
from openagentflow.distributed.kubernetes import KubernetesBackend
from openagentflow.distributed.provider import DistributedOllamaProvider
from openagentflow.distributed.ssh import SSHBackend

__all__ = [
    # Core types
    "ComputeBackend",
    "ComputeBackendInterface",
    "ComputeCluster",
    "ComputeNode",
    # Load balancers
    "LoadBalancer",
    "RoundRobinBalancer",
    "LeastLoadBalancer",
    "ModelAffinityBalancer",
    # Backends
    "KubernetesBackend",
    "DockerBackend",
    "SSHBackend",
    # Distributed provider
    "DistributedOllamaProvider",
]
