"""Distributed Ollama provider with load balancing and automatic failover.

Wraps the regular :class:`OllamaProvider` with cluster awareness.  Requests
are transparently routed to the best available node in a
:class:`ComputeCluster`, and failures automatically trigger retries on
alternate nodes.

This is the primary user-facing class for distributed inference.

Example::

    from openagentflow.distributed import (
        ComputeBackend,
        ComputeCluster,
        ComputeNode,
        DistributedOllamaProvider,
    )

    cluster = ComputeCluster("gpu-pool")
    cluster.add_node(
        ComputeNode("gpu1", ComputeBackend.HTTP, "http://gpu1:11434", model="llama3")
    )
    cluster.add_node(
        ComputeNode("gpu2", ComputeBackend.HTTP, "http://gpu2:11434", model="llama3")
    )

    provider = DistributedOllamaProvider(cluster=cluster)

    # Uses the best available node, with automatic failover
    result = await provider.generate(
        messages=[Message(role="user", content="Hello!")],
        config=ModelConfig(model_id="llama3"),
    )
    print(result.content)

    # Auto-discover nodes from a backend
    from openagentflow.distributed.kubernetes import KubernetesBackend
    k8s = KubernetesBackend(namespace="ml")
    await provider.auto_discover(k8s)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator

from openagentflow.distributed.base import (
    ComputeBackendInterface,
    ComputeCluster,
    ComputeNode,
    LeastLoadBalancer,
    LoadBalancer,
)
from openagentflow.exceptions import LLMError
from openagentflow.llm.base import BaseLLMProvider, LLMResponse, StreamChunk

if TYPE_CHECKING:
    from openagentflow.core.types import Message, ModelConfig, ToolSpec

logger = logging.getLogger(__name__)


class DistributedOllamaProvider(BaseLLMProvider):
    """A distributed Ollama provider that load-balances across a cluster.

    Routes each inference request to the best available
    :class:`ComputeNode` in the cluster, using a pluggable
    :class:`LoadBalancer` strategy.  On failure the request is
    automatically retried on a different node (up to *max_retries* times).

    Background health checks can be started with
    :meth:`start_health_checks` to continuously monitor node availability.

    Args:
        cluster:                The compute cluster to distribute across.
        balancer:               Load-balancing strategy (defaults to
                                :class:`LeastLoadBalancer`).
        health_check_interval:  Seconds between background health checks.
        max_retries:            Maximum number of failover retries per request.
        recovery_interval:      Seconds before retrying an unhealthy node
                                during background health checks.
    """

    def __init__(
        self,
        cluster: ComputeCluster,
        balancer: LoadBalancer | None = None,
        health_check_interval: int = 30,
        max_retries: int = 2,
        recovery_interval: int = 60,
    ) -> None:
        self.cluster = cluster
        self.balancer = balancer or LeastLoadBalancer()
        self.health_check_interval = health_check_interval
        self.max_retries = max_retries
        self.recovery_interval = recovery_interval

        self._health_check_task: asyncio.Task[None] | None = None
        self._backends: list[ComputeBackendInterface] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_node(self, model: str | None = None) -> ComputeNode:
        """Select a node via the balancer or raise if none available.

        Args:
            model: Requested model name.

        Returns:
            Selected :class:`ComputeNode`.

        Raises:
            LLMError: If no node is available.
        """
        node = self.balancer.select_node(self.cluster, model=model)
        if node is None:
            available = self.cluster.get_available_nodes(model)
            total = self.cluster.total_nodes
            raise LLMError(
                f"No available Ollama nodes in cluster '{self.cluster.name}' "
                f"for model '{model}'. "
                f"Total nodes: {total}, available: {len(available)}. "
                f"Check that nodes are healthy and have capacity."
            )
        return node

    def _make_ollama_provider(
        self, node: ComputeNode
    ) -> "OllamaProvider":
        """Create an OllamaProvider pointing at a specific node.

        Args:
            node: The node to target.

        Returns:
            Configured :class:`OllamaProvider` instance.
        """
        from openagentflow.llm.providers.ollama_ import OllamaProvider

        return OllamaProvider(base_url=node.endpoint)

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response, routing to the best available node.

        The request is retried on a different node if the selected node fails
        (up to ``self.max_retries`` times).

        Args:
            messages:      Conversation history.
            config:        Model configuration.
            tools:         Available tools for function calling.
            system_prompt: Optional system prompt.

        Returns:
            LLMResponse from the successful node.

        Raises:
            LLMError: If all retry attempts are exhausted.
        """
        model = getattr(config, "model_id", None)
        tried_nodes: set[str] = set()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                node = self._select_node(model)

                # Avoid retrying the same node immediately
                if node.node_id in tried_nodes:
                    # Try to find a different node
                    alternatives = [
                        n for n in self.cluster.get_available_nodes(model)
                        if n.node_id not in tried_nodes
                    ]
                    if alternatives:
                        node = alternatives[0]
                    elif attempt > 0:
                        # All nodes tried -- give the original another shot
                        pass

                tried_nodes.add(node.node_id)
                node.increment_load()

                try:
                    provider = self._make_ollama_provider(node)
                    response = await provider.generate(
                        messages=messages,
                        config=config,
                        tools=tools,
                        system_prompt=system_prompt,
                    )
                    logger.debug(
                        "Request served by node %s (attempt %d)",
                        node.node_id,
                        attempt + 1,
                    )
                    return response

                except Exception as exc:
                    last_error = exc
                    node.mark_unhealthy()
                    logger.warning(
                        "Node %s failed (attempt %d/%d): %s",
                        node.node_id,
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                    )

                finally:
                    node.decrement_load()

            except LLMError:
                # No node available at all
                if attempt < self.max_retries:
                    await asyncio.sleep(1)  # Brief pause before retry
                    continue
                raise

        raise LLMError(
            f"All {self.max_retries + 1} attempts failed across cluster "
            f"'{self.cluster.name}'. Last error: {last_error}"
        )

    async def generate_stream(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response with automatic node selection.

        Unlike :meth:`generate`, streaming does not support mid-stream
        failover -- if the selected node fails after streaming has started,
        the error is propagated to the caller.  However, the initial node
        selection still uses the load balancer, and a failed connection
        attempt triggers a retry on a different node.

        Args:
            messages:      Conversation history.
            config:        Model configuration.
            tools:         Available tools.
            system_prompt: Optional system prompt.

        Yields:
            :class:`StreamChunk` objects.

        Raises:
            LLMError: If the initial connection fails on all retried nodes.
        """
        model = getattr(config, "model_id", None)
        tried_nodes: set[str] = set()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            node = self._select_node(model)

            if node.node_id in tried_nodes:
                alternatives = [
                    n for n in self.cluster.get_available_nodes(model)
                    if n.node_id not in tried_nodes
                ]
                if alternatives:
                    node = alternatives[0]

            tried_nodes.add(node.node_id)
            node.increment_load()

            try:
                provider = self._make_ollama_provider(node)
                async for chunk in provider.generate_stream(
                    messages=messages,
                    config=config,
                    tools=tools,
                    system_prompt=system_prompt,
                ):
                    yield chunk

                # Stream completed successfully
                return

            except LLMError as exc:
                last_error = exc
                node.mark_unhealthy()
                logger.warning(
                    "Streaming node %s failed (attempt %d/%d): %s",
                    node.node_id,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt >= self.max_retries:
                    raise

            finally:
                node.decrement_load()

        raise LLMError(
            f"All streaming attempts failed across cluster "
            f"'{self.cluster.name}'. Last error: {last_error}"
        )

    def count_tokens(self, text: str, model_id: str) -> int:
        """Count tokens using any available healthy node.

        Falls back to a heuristic (~4 chars / token) if no node is available.

        Args:
            text:     Text to tokenize.
            model_id: Model for tokenization context.

        Returns:
            Token count.
        """
        available = self.cluster.get_available_nodes()
        if available:
            provider = self._make_ollama_provider(available[0])
            try:
                return provider.count_tokens(text, model_id)
            except Exception:
                pass

        # Fallback heuristic
        return len(text) // 4

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model_id: str
    ) -> float:
        """Estimate cost.  Local Ollama is always 0.0."""
        return 0.0

    @property
    def provider_name(self) -> str:
        return "distributed-ollama"

    @property
    def supported_models(self) -> list[str]:
        """Aggregate supported models from all nodes."""
        models: set[str] = set()
        for node in self.cluster.nodes:
            if node.model:
                models.add(node.model)
        return sorted(models) if models else ["*"]

    def supports_model(self, model_id: str) -> bool:
        """Ollama supports any pulled model."""
        return True

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    async def run_health_checks(self) -> dict[str, bool]:
        """Run a one-shot health check on all nodes in the cluster.

        For each node, performs an HTTP health check.  Nodes whose backend
        has a registered :class:`ComputeBackendInterface` use that backend's
        ``health_check`` method; otherwise a direct HTTP GET to
        ``/api/tags`` is attempted.

        Returns:
            Dict mapping ``node_id`` -> health status.
        """
        results: dict[str, bool] = {}

        for node in self.cluster.nodes:
            try:
                healthy = await self._check_node_health(node)
                results[node.node_id] = healthy
            except Exception as exc:
                logger.debug("Health check error for %s: %s", node.node_id, exc)
                node.mark_unhealthy()
                results[node.node_id] = False

        healthy_count = sum(1 for v in results.values() if v)
        logger.info(
            "Health check complete: %d/%d nodes healthy in cluster '%s'",
            healthy_count,
            len(results),
            self.cluster.name,
        )
        return results

    async def _check_node_health(self, node: ComputeNode) -> bool:
        """Check a single node's health.

        Tries registered backends first, then falls back to a direct HTTP
        check.
        """
        # Try registered backends
        for backend in self._backends:
            try:
                result = await backend.health_check(node)
                return result
            except Exception:
                continue

        # Direct HTTP fallback
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(None, self._direct_health_check, node)
        if ok:
            node.mark_healthy()
        else:
            node.mark_unhealthy()
        return ok

    @staticmethod
    def _direct_health_check(node: ComputeNode) -> bool:
        """Direct HTTP health check to Ollama's /api/tags endpoint."""
        import urllib.request

        url = f"{node.endpoint.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            return resp.status == 200
        except Exception:
            return False

    async def start_health_checks(self) -> None:
        """Start periodic background health checks.

        Health checks run every ``self.health_check_interval`` seconds in an
        ``asyncio.Task``.  Call :meth:`stop_health_checks` to cancel.
        """
        if self._health_check_task and not self._health_check_task.done():
            logger.warning("Health check loop is already running")
            return

        self._health_check_task = asyncio.create_task(
            self._health_check_loop(), name="distributed-ollama-health-checks"
        )
        logger.info(
            "Started background health checks every %ds for cluster '%s'",
            self.health_check_interval,
            self.cluster.name,
        )

    async def stop_health_checks(self) -> None:
        """Stop the background health check loop."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped background health checks for cluster '%s'", self.cluster.name)

    async def _health_check_loop(self) -> None:
        """Background loop that periodically runs health checks."""
        while True:
            try:
                await self.run_health_checks()
            except Exception as exc:
                logger.error("Health check loop error: %s", exc)

            await asyncio.sleep(self.health_check_interval)

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------

    async def auto_discover(
        self, backend: ComputeBackendInterface, *, replace: bool = False
    ) -> list[ComputeNode]:
        """Auto-discover nodes from a backend and add them to the cluster.

        Args:
            backend:  A backend instance (:class:`KubernetesBackend`,
                      :class:`DockerBackend`, or :class:`SSHBackend`).
            replace:  If ``True``, remove existing nodes from the same backend
                      type before adding discovered nodes.

        Returns:
            List of newly discovered :class:`ComputeNode` instances.
        """
        discovered = await backend.discover_nodes()

        if replace:
            # Remove existing nodes from this backend type
            existing_ids = [
                n.node_id
                for n in self.cluster.nodes
                if n.backend == discovered[0].backend
            ] if discovered else []
            for node_id in existing_ids:
                self.cluster.remove_node(node_id)

        for node in discovered:
            self.cluster.add_node(node)

        # Register the backend for health checks
        if backend not in self._backends:
            self._backends.append(backend)

        logger.info(
            "Auto-discovered %d nodes from %s backend into cluster '%s'",
            len(discovered),
            type(backend).__name__,
            self.cluster.name,
        )
        return discovered

    # ------------------------------------------------------------------
    # Cluster info / convenience
    # ------------------------------------------------------------------

    def get_cluster_status(self) -> dict[str, Any]:
        """Return a summary of the cluster state.

        Returns:
            Dict with ``cluster_name``, ``total_nodes``, ``healthy_nodes``,
            ``total_capacity``, ``current_load``, and per-node details.
        """
        nodes_info = []
        for node in self.cluster.nodes:
            nodes_info.append({
                "node_id": node.node_id,
                "backend": node.backend.value,
                "endpoint": node.endpoint,
                "model": node.model,
                "healthy": node.healthy,
                "load": f"{node.current_load}/{node.max_concurrent}",
                "gpu_memory_mb": node.gpu_memory_mb,
            })

        return {
            "cluster_name": self.cluster.name,
            "total_nodes": self.cluster.total_nodes,
            "healthy_nodes": self.cluster.healthy_nodes,
            "total_capacity": self.cluster.total_capacity,
            "current_load": self.cluster.current_total_load,
            "balancer": type(self.balancer).__name__,
            "nodes": nodes_info,
        }
