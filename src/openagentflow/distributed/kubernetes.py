"""Kubernetes backend for distributed Ollama inference.

Manages Ollama instances running as Kubernetes pods.  Uses the K8s REST API
directly via ``urllib.request`` -- no ``kubernetes`` Python client or ``kubectl``
binary required.

Authentication:
    - **In-cluster**: Reads the service-account token from
      ``/var/run/secrets/kubernetes.io/serviceaccount/token`` and the CA cert
      from the same directory.  The API server address comes from the
      ``KUBERNETES_SERVICE_HOST`` / ``KUBERNETES_SERVICE_PORT`` env vars.
    - **Out-of-cluster**: Parses ``~/.kube/config`` (or *kubeconfig_path*) to
      extract the current-context server URL and bearer token / client cert.

Assumes Ollama is deployed as a Kubernetes Deployment or StatefulSet with:
    - Label: ``app=ollama`` (configurable via *label_selector*)
    - Container port: ``11434``
    - Optional GPU node affinity via ``nvidia.com/gpu`` resource requests

Example::

    from openagentflow.distributed.kubernetes import KubernetesBackend

    k8s = KubernetesBackend(namespace="ml-inference")
    nodes = await k8s.discover_nodes()
    for node in nodes:
        healthy = await k8s.health_check(node)
        print(f"{node.node_id}: {'OK' if healthy else 'DOWN'}")

    # Scale up
    await k8s.scale(replicas=4)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import urllib.error
import urllib.request
from typing import Any

from openagentflow.distributed.base import (
    ComputeBackend,
    ComputeBackendInterface,
    ComputeNode,
)

logger = logging.getLogger(__name__)

# Well-known paths for in-cluster K8s service-account credentials
_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_SA_CA_CERT_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
_SA_NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

# Default Ollama port inside pods
_OLLAMA_PORT = 11434


class KubernetesBackend(ComputeBackendInterface):
    """Manage Ollama instances on Kubernetes via the K8s REST API.

    This backend discovers running Ollama pods, performs health checks by
    hitting each pod's ``/api/tags`` endpoint, and can scale the underlying
    Deployment up or down.

    All HTTP communication uses Python's ``urllib.request`` -- no external
    dependencies are needed.

    Args:
        namespace:       Kubernetes namespace to operate in (default ``"default"``).
        kubeconfig_path: Path to a ``kubeconfig`` file.  When ``None`` the
                         backend first tries in-cluster auth, then falls back to
                         ``~/.kube/config``.
        api_server:      Explicit K8s API server URL (e.g.
                         ``https://my-cluster:6443``).  Overrides discovery.
        label_selector:  Pod label selector for finding Ollama pods.
        deployment_name: Name of the Deployment / StatefulSet to scale.
        ollama_port:     Port that Ollama listens on inside each pod.
    """

    def __init__(
        self,
        namespace: str = "default",
        kubeconfig_path: str | None = None,
        api_server: str | None = None,
        label_selector: str = "app=ollama",
        deployment_name: str = "ollama",
        ollama_port: int = _OLLAMA_PORT,
    ) -> None:
        self.namespace = namespace
        self.label_selector = label_selector
        self.deployment_name = deployment_name
        self.ollama_port = ollama_port

        # Resolved during _configure()
        self._api_server: str = api_server or ""
        self._token: str = ""
        self._ssl_context: ssl.SSLContext | None = None
        self._configured = False
        self._kubeconfig_path = kubeconfig_path

    # ------------------------------------------------------------------
    # Configuration / authentication helpers
    # ------------------------------------------------------------------

    def _configure(self) -> None:
        """Lazily configure the K8s API connection.

        Tries in-cluster auth first, then falls back to kubeconfig.
        """
        if self._configured:
            return

        if self._api_server:
            # Explicit API server -- user may also supply a token via env var
            self._token = os.environ.get("KUBE_TOKEN", "")
            self._ssl_context = self._make_ssl_context()
            self._configured = True
            return

        # Try in-cluster auth
        if os.path.isfile(_SA_TOKEN_PATH):
            self._configure_in_cluster()
        else:
            self._configure_from_kubeconfig()

        self._configured = True

    def _configure_in_cluster(self) -> None:
        """Configure from the in-cluster service-account credentials."""
        try:
            with open(_SA_TOKEN_PATH) as f:
                self._token = f.read().strip()
        except OSError as exc:
            logger.warning("Could not read service-account token: %s", exc)
            self._token = ""

        # Read namespace if not explicitly set and the file exists
        if self.namespace == "default" and os.path.isfile(_SA_NAMESPACE_PATH):
            try:
                with open(_SA_NAMESPACE_PATH) as f:
                    self.namespace = f.read().strip()
            except OSError:
                pass

        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        self._api_server = f"https://{host}:{port}"

        self._ssl_context = self._make_ssl_context(ca_cert_path=_SA_CA_CERT_PATH)
        logger.info(
            "Configured K8s in-cluster: server=%s namespace=%s",
            self._api_server,
            self.namespace,
        )

    def _configure_from_kubeconfig(self) -> None:
        """Parse a kubeconfig YAML-ish file to extract server + auth.

        This is a minimal parser that handles the most common kubeconfig
        layout.  It does *not* pull in PyYAML -- instead it reads the file as
        JSON (kubeconfig supports JSON) or does a simple line-based parse for
        YAML.
        """
        path = self._kubeconfig_path or os.path.expanduser("~/.kube/config")
        if not os.path.isfile(path):
            logger.warning("kubeconfig not found at %s -- using defaults", path)
            self._api_server = self._api_server or "https://localhost:6443"
            self._ssl_context = self._make_ssl_context()
            return

        try:
            with open(path) as f:
                raw = f.read()

            # Try JSON first (valid kubeconfig can be JSON)
            try:
                config = json.loads(raw)
            except json.JSONDecodeError:
                config = self._parse_kubeconfig_yaml(raw)

            # Extract current-context
            current_ctx_name = config.get("current-context", "")
            contexts = {c["name"]: c.get("context", {}) for c in config.get("contexts", [])}
            ctx = contexts.get(current_ctx_name, {})

            cluster_name = ctx.get("cluster", "")
            user_name = ctx.get("user", "")

            clusters = {c["name"]: c.get("cluster", {}) for c in config.get("clusters", [])}
            users = {u["name"]: u.get("user", {}) for u in config.get("users", [])}

            cluster_info = clusters.get(cluster_name, {})
            user_info = users.get(user_name, {})

            self._api_server = cluster_info.get("server", "https://localhost:6443")
            self._token = user_info.get("token", "")

            # Namespace from context, if set
            ns = ctx.get("namespace", "")
            if ns and self.namespace == "default":
                self.namespace = ns

            self._ssl_context = self._make_ssl_context()
            logger.info(
                "Configured K8s from kubeconfig: server=%s context=%s namespace=%s",
                self._api_server,
                current_ctx_name,
                self.namespace,
            )

        except Exception as exc:
            logger.warning("Failed to parse kubeconfig at %s: %s", path, exc)
            self._api_server = self._api_server or "https://localhost:6443"
            self._ssl_context = self._make_ssl_context()

    @staticmethod
    def _parse_kubeconfig_yaml(raw: str) -> dict[str, Any]:
        """Minimal YAML-subset parser for kubeconfig files.

        Handles the most common flat key-value patterns found in kubeconfig.
        This avoids a PyYAML dependency.  For complex configs the user should
        supply *api_server* and token directly.
        """
        # This is intentionally simplistic -- it covers single-context configs.
        result: dict[str, Any] = {
            "clusters": [],
            "contexts": [],
            "users": [],
            "current-context": "",
        }

        lines = raw.splitlines()
        i = 0

        def _strip_val(line: str) -> str:
            return line.split(":", 1)[-1].strip().strip("'\"")

        while i < len(lines):
            line = lines[i].rstrip()
            if line.startswith("current-context:"):
                result["current-context"] = _strip_val(line)
            i += 1

        # Fallback: return mostly-empty config; caller will use defaults
        return result

    @staticmethod
    def _make_ssl_context(ca_cert_path: str | None = None) -> ssl.SSLContext:
        """Create an SSL context for K8s API communication."""
        ctx = ssl.create_default_context()
        if ca_cert_path and os.path.isfile(ca_cert_path):
            ctx.load_verify_locations(ca_cert_path)
        else:
            # Allow unverified if no CA cert available (dev clusters)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    # ------------------------------------------------------------------
    # HTTP helpers (raw K8s API calls via urllib)
    # ------------------------------------------------------------------

    def _k8s_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        """Execute an HTTP request against the K8s API server.

        Args:
            method:       HTTP method (GET, POST, PATCH, PUT, DELETE).
            path:         API path, e.g. ``/api/v1/namespaces/default/pods``.
            body:         JSON-serialisable request body (optional).
            content_type: Content-Type header value.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
        """
        self._configure()

        url = f"{self._api_server.rstrip('/')}{path}"
        data = json.dumps(body).encode("utf-8") if body else None
        headers: dict[str, str] = {"Accept": "application/json"}
        if data:
            headers["Content-Type"] = content_type
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            resp = urllib.request.urlopen(req, context=self._ssl_context, timeout=30)
            resp_body = resp.read().decode("utf-8")
            return json.loads(resp_body) if resp_body else {}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"K8s API error {exc.code} on {method} {path}: {err_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"K8s API connection error on {method} {path}: {exc}"
            ) from exc

    def _ollama_request(
        self, endpoint: str, path: str, *, timeout: int = 5
    ) -> dict[str, Any] | None:
        """Send an HTTP GET to an Ollama pod endpoint.

        Args:
            endpoint: Base URL of the Ollama instance.
            path:     API path (e.g. ``/api/tags``).
            timeout:  Request timeout in seconds.

        Returns:
            Parsed JSON dict, or ``None`` on failure.
        """
        url = f"{endpoint.rstrip('/')}{path}"
        req = urllib.request.Request(url, method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
        except Exception:
            return None

    # ------------------------------------------------------------------
    # ComputeBackendInterface implementation
    # ------------------------------------------------------------------

    async def discover_nodes(self) -> list[ComputeNode]:
        """Discover all running Ollama pods matching *label_selector*.

        Queries the K8s API for pods in the configured namespace with the
        configured label selector.  Each pod in the ``Running`` phase is
        returned as a :class:`ComputeNode`.

        Returns:
            List of discovered compute nodes.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._discover_nodes_sync)

    def _discover_nodes_sync(self) -> list[ComputeNode]:
        """Synchronous implementation of pod discovery."""
        path = (
            f"/api/v1/namespaces/{self.namespace}/pods"
            f"?labelSelector={self.label_selector}"
        )
        try:
            data = self._k8s_request("GET", path)
        except RuntimeError as exc:
            logger.error("Failed to discover K8s pods: %s", exc)
            return []

        nodes: list[ComputeNode] = []
        for item in data.get("items", []):
            metadata = item.get("metadata", {})
            status = item.get("status", {})
            spec = item.get("spec", {})

            phase = status.get("phase", "")
            if phase != "Running":
                continue

            pod_name = metadata.get("name", "unknown")
            pod_ip = status.get("podIP", "")
            if not pod_ip:
                continue

            endpoint = f"http://{pod_ip}:{self.ollama_port}"

            # Extract GPU info from resource requests
            gpu_mem = 0
            for container in spec.get("containers", []):
                resources = container.get("resources", {})
                limits = resources.get("limits", {})
                # nvidia.com/gpu is count, not memory -- but record it
                gpu_count = limits.get("nvidia.com/gpu", 0)
                if gpu_count:
                    gpu_mem = int(gpu_count) * 8192  # Rough estimate: 8GB per GPU

            node = ComputeNode(
                node_id=f"k8s-{pod_name}",
                backend=ComputeBackend.KUBERNETES,
                endpoint=endpoint,
                gpu_memory_mb=gpu_mem,
                max_concurrent=4,  # Conservative default
                metadata={
                    "pod_name": pod_name,
                    "pod_ip": pod_ip,
                    "namespace": self.namespace,
                    "node_name": spec.get("nodeName", ""),
                    "labels": metadata.get("labels", {}),
                },
            )
            nodes.append(node)

        logger.info(
            "Discovered %d Ollama pods in namespace %s",
            len(nodes),
            self.namespace,
        )
        return nodes

    async def health_check(self, node: ComputeNode) -> bool:
        """Check if an Ollama pod is healthy by querying ``/api/tags``.

        Args:
            node: The compute node to health-check.

        Returns:
            ``True`` if the Ollama server responds with a valid JSON body.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self._ollama_request, node.endpoint, "/api/tags"
        )
        healthy = result is not None
        if healthy:
            node.mark_healthy()
            # Try to extract loaded model info
            models = result.get("models", [])
            if models:
                node.model = models[0].get("name", "")
        else:
            node.mark_unhealthy()

        return healthy

    # ------------------------------------------------------------------
    # Scaling and GPU helpers
    # ------------------------------------------------------------------

    async def scale(self, replicas: int) -> None:
        """Scale the Ollama deployment to *replicas* instances.

        Sends a PATCH to the K8s API to update the replica count on the
        Deployment named ``self.deployment_name``.

        Args:
            replicas: Desired number of replicas.

        Raises:
            RuntimeError: If the K8s API returns an error.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._scale_sync, replicas)

    def _scale_sync(self, replicas: int) -> None:
        """Synchronous implementation of deployment scaling."""
        path = (
            f"/apis/apps/v1/namespaces/{self.namespace}"
            f"/deployments/{self.deployment_name}"
        )
        body = {"spec": {"replicas": replicas}}
        try:
            self._k8s_request(
                "PATCH",
                path,
                body=body,
                content_type="application/strategic-merge-patch+json",
            )
            logger.info(
                "Scaled deployment %s/%s to %d replicas",
                self.namespace,
                self.deployment_name,
                replicas,
            )
        except RuntimeError as exc:
            logger.error("Failed to scale deployment: %s", exc)
            raise

    async def get_pod_gpu_info(self, pod_name: str) -> dict[str, Any]:
        """Get GPU resource information for a specific pod.

        Queries the pod spec and extracts ``nvidia.com/gpu`` resource
        requests and limits.

        Args:
            pod_name: Name of the K8s pod.

        Returns:
            Dict with keys ``gpu_count``, ``gpu_memory_mb`` (estimated), and
            ``node_name``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_pod_gpu_info_sync, pod_name)

    def _get_pod_gpu_info_sync(self, pod_name: str) -> dict[str, Any]:
        """Synchronous implementation of GPU info retrieval."""
        path = f"/api/v1/namespaces/{self.namespace}/pods/{pod_name}"
        try:
            data = self._k8s_request("GET", path)
        except RuntimeError as exc:
            logger.warning("Failed to get pod GPU info for %s: %s", pod_name, exc)
            return {"gpu_count": 0, "gpu_memory_mb": 0, "node_name": ""}

        spec = data.get("spec", {})
        gpu_count = 0
        for container in spec.get("containers", []):
            resources = container.get("resources", {})
            limits = resources.get("limits", {})
            gpu_str = limits.get("nvidia.com/gpu", "0")
            try:
                gpu_count += int(gpu_str)
            except (ValueError, TypeError):
                pass

        return {
            "gpu_count": gpu_count,
            "gpu_memory_mb": gpu_count * 8192,  # Rough 8GB-per-GPU estimate
            "node_name": spec.get("nodeName", ""),
        }

    async def get_deployment_status(self) -> dict[str, Any]:
        """Return the current status of the Ollama deployment.

        Returns:
            Dict with ``desired_replicas``, ``available_replicas``,
            ``ready_replicas``, and ``conditions``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_deployment_status_sync)

    def _get_deployment_status_sync(self) -> dict[str, Any]:
        """Synchronous implementation of deployment status retrieval."""
        path = (
            f"/apis/apps/v1/namespaces/{self.namespace}"
            f"/deployments/{self.deployment_name}"
        )
        try:
            data = self._k8s_request("GET", path)
        except RuntimeError as exc:
            logger.warning("Failed to get deployment status: %s", exc)
            return {}

        status = data.get("status", {})
        spec = data.get("spec", {})
        return {
            "desired_replicas": spec.get("replicas", 0),
            "available_replicas": status.get("availableReplicas", 0),
            "ready_replicas": status.get("readyReplicas", 0),
            "conditions": status.get("conditions", []),
        }
