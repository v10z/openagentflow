"""Docker backend for distributed Ollama inference.

Manages Ollama instances running as Docker containers.  Communicates with the
Docker Engine API over HTTP -- either via a Unix socket
(``unix:///var/run/docker.sock``) or a TCP endpoint
(``tcp://docker-host:2375``).

No ``docker-py`` or ``docker`` CLI dependency is required; all communication
uses Python's ``urllib.request``.

Example::

    from openagentflow.distributed.docker import DockerBackend

    docker = DockerBackend()
    nodes = await docker.discover_nodes()

    # Start a new Ollama container with GPU support
    new_node = await docker.start_node(model="llama3", gpu_devices="all")

    # Health-check it
    ok = await docker.health_check(new_node)
"""

from __future__ import annotations

import asyncio
import http.client
import json
import logging
import platform
import socket
import urllib.error
import urllib.request
from typing import Any
from uuid import uuid4

from openagentflow.distributed.base import (
    ComputeBackend,
    ComputeBackendInterface,
    ComputeNode,
)

logger = logging.getLogger(__name__)

# Default Ollama image and port
_DEFAULT_IMAGE = "ollama/ollama"
_OLLAMA_PORT = 11434


# ---------------------------------------------------------------------------
# Unix-socket HTTP support for Docker Engine API
# ---------------------------------------------------------------------------

class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection subclass that connects via a Unix domain socket.

    Used to talk to the Docker daemon through
    ``/var/run/docker.sock`` on Linux / macOS.
    """

    def __init__(self, socket_path: str, timeout: int = 30) -> None:
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:  # noqa: D401
        """Override connect to use a Unix socket."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


class DockerBackend(ComputeBackendInterface):
    """Manage Ollama instances via the Docker Engine API.

    Supports discovering running Ollama containers, starting new ones (with
    optional GPU passthrough), stopping containers, and performing health
    checks.

    On Linux and macOS the default ``docker_host`` points at the Unix socket.
    On Windows (or when Docker is exposed over TCP) supply a TCP URL instead.

    Args:
        docker_host: Docker Engine API endpoint.  Accepted formats:
                     ``unix:///var/run/docker.sock`` (default on Linux/macOS),
                     ``tcp://hostname:2375``, ``http://hostname:2375``.
        image:       Docker image to use when starting new Ollama containers.
        gpu_enabled: Whether to request GPU access for new containers.
        ollama_port: Port Ollama listens on inside the container.
        container_prefix: Prefix for container names created by this backend.
    """

    def __init__(
        self,
        docker_host: str | None = None,
        image: str = _DEFAULT_IMAGE,
        gpu_enabled: bool = True,
        ollama_port: int = _OLLAMA_PORT,
        container_prefix: str = "oaf-ollama",
    ) -> None:
        if docker_host is None:
            if platform.system() == "Windows":
                docker_host = "tcp://localhost:2375"
            else:
                docker_host = "unix:///var/run/docker.sock"

        self._docker_host = docker_host
        self.image = image
        self.gpu_enabled = gpu_enabled
        self.ollama_port = ollama_port
        self.container_prefix = container_prefix

    # ------------------------------------------------------------------
    # Docker Engine API helpers
    # ------------------------------------------------------------------

    def _docker_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict[str, Any] | list[Any]:
        """Send a request to the Docker Engine API.

        Handles both Unix-socket and TCP connections transparently.

        Args:
            method:  HTTP method.
            path:    API path (e.g. ``/containers/json``).
            body:    JSON-serialisable request body.
            timeout: Request timeout in seconds.

        Returns:
            Parsed JSON response (dict or list).

        Raises:
            RuntimeError: On communication or HTTP errors.
        """
        data = json.dumps(body).encode("utf-8") if body else None
        headers: dict[str, str] = {"Content-Type": "application/json"}

        if self._docker_host.startswith("unix://"):
            return self._docker_unix_request(method, path, data, headers, timeout)
        else:
            return self._docker_tcp_request(method, path, data, headers, timeout)

    def _docker_unix_request(
        self,
        method: str,
        path: str,
        data: bytes | None,
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any] | list[Any]:
        """Execute a Docker API request over a Unix socket."""
        socket_path = self._docker_host.replace("unix://", "")
        conn = _UnixHTTPConnection(socket_path, timeout=timeout)
        try:
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read().decode("utf-8")
            if resp.status >= 400:
                raise RuntimeError(
                    f"Docker API error {resp.status} on {method} {path}: {resp_body}"
                )
            return json.loads(resp_body) if resp_body else {}
        except OSError as exc:
            raise RuntimeError(
                f"Docker socket error ({socket_path}): {exc}"
            ) from exc
        finally:
            conn.close()

    def _docker_tcp_request(
        self,
        method: str,
        path: str,
        data: bytes | None,
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any] | list[Any]:
        """Execute a Docker API request over TCP / HTTP."""
        # Normalise host URL
        host_url = self._docker_host
        for prefix in ("tcp://", "http://"):
            if host_url.startswith(prefix):
                host_url = "http://" + host_url[len(prefix):]
                break

        url = f"{host_url.rstrip('/')}{path}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Docker API error {exc.code} on {method} {path}: {err_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Docker API connection error on {method} {path}: {exc}"
            ) from exc

    def _ollama_health_request(self, endpoint: str, timeout: int = 5) -> bool:
        """Send a GET to an Ollama endpoint's ``/api/tags`` to check health."""
        url = f"{endpoint.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            return resp.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # ComputeBackendInterface implementation
    # ------------------------------------------------------------------

    async def discover_nodes(self) -> list[ComputeNode]:
        """Find all running Ollama containers.

        Queries the Docker API for containers whose image matches
        ``self.image`` (with or without a tag suffix) and that are in the
        ``running`` state.

        Returns:
            List of discovered :class:`ComputeNode` instances.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._discover_nodes_sync)

    def _discover_nodes_sync(self) -> list[ComputeNode]:
        """Synchronous container discovery."""
        try:
            containers = self._docker_request("GET", "/containers/json")
        except RuntimeError as exc:
            logger.error("Failed to list Docker containers: %s", exc)
            return []

        if not isinstance(containers, list):
            return []

        nodes: list[ComputeNode] = []
        for container in containers:
            image = container.get("Image", "")
            # Match on image name (with or without :tag)
            image_base = image.split(":")[0]
            if self.image.split(":")[0] not in image_base:
                continue

            container_id = container.get("Id", "")[:12]
            names = container.get("Names", [])
            name = names[0].lstrip("/") if names else container_id
            state = container.get("State", "")
            if state != "running":
                continue

            # Determine the host-accessible endpoint
            ports = container.get("Ports", [])
            host_port = self._find_host_port(ports, self.ollama_port)
            if host_port:
                endpoint = f"http://localhost:{host_port}"
            else:
                # Try to get the container's IP on the bridge network
                networks = container.get("NetworkSettings", {}).get("Networks", {})
                ip = ""
                for net_info in networks.values():
                    ip = net_info.get("IPAddress", "")
                    if ip:
                        break
                if ip:
                    endpoint = f"http://{ip}:{self.ollama_port}"
                else:
                    endpoint = f"http://localhost:{self.ollama_port}"

            node = ComputeNode(
                node_id=f"docker-{container_id}",
                backend=ComputeBackend.DOCKER,
                endpoint=endpoint,
                max_concurrent=4,
                metadata={
                    "container_id": container.get("Id", ""),
                    "container_name": name,
                    "image": image,
                    "created": container.get("Created", 0),
                },
            )
            nodes.append(node)

        logger.info("Discovered %d Ollama Docker containers", len(nodes))
        return nodes

    @staticmethod
    def _find_host_port(ports: list[dict[str, Any]], private_port: int) -> int | None:
        """Find the host port mapped to *private_port* in a Docker port list."""
        for mapping in ports:
            if mapping.get("PrivatePort") == private_port and mapping.get("PublicPort"):
                return mapping["PublicPort"]
        return None

    async def health_check(self, node: ComputeNode) -> bool:
        """Check if an Ollama container is healthy via ``/api/tags``.

        Args:
            node: The node to health-check.

        Returns:
            ``True`` if the Ollama server inside the container responds.
        """
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(
            None, self._ollama_health_request, node.endpoint
        )
        if ok:
            node.mark_healthy()
        else:
            node.mark_unhealthy()
        return ok

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    async def start_node(
        self,
        model: str | None = None,
        gpu_devices: str = "all",
        host_port: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> ComputeNode:
        """Start a new Ollama container with optional GPU access.

        Args:
            model:       Model to pre-pull after the container starts.
            gpu_devices: GPU device request (``"all"`` or comma-separated IDs).
                         Only used when ``self.gpu_enabled`` is ``True``.
            host_port:   Host port to bind for the Ollama API.  ``None`` means
                         Docker assigns a random port.
            extra_env:   Additional environment variables for the container.

        Returns:
            A :class:`ComputeNode` representing the newly started container.

        Raises:
            RuntimeError: If the Docker API returns an error.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._start_node_sync,
            model,
            gpu_devices,
            host_port,
            extra_env,
        )

    def _start_node_sync(
        self,
        model: str | None,
        gpu_devices: str,
        host_port: int | None,
        extra_env: dict[str, str] | None,
    ) -> ComputeNode:
        """Synchronous implementation of container start."""
        short_id = str(uuid4())[:8]
        container_name = f"{self.container_prefix}-{short_id}"

        # Build container creation payload
        env_list = [f"{k}={v}" for k, v in (extra_env or {}).items()]

        host_config: dict[str, Any] = {}
        if host_port:
            host_config["PortBindings"] = {
                f"{self.ollama_port}/tcp": [{"HostPort": str(host_port)}]
            }
        else:
            host_config["PublishAllPorts"] = True

        if self.gpu_enabled:
            host_config["DeviceRequests"] = [
                {
                    "Driver": "nvidia",
                    "Count": -1 if gpu_devices == "all" else 0,
                    "DeviceIDs": [] if gpu_devices == "all" else gpu_devices.split(","),
                    "Capabilities": [["gpu"]],
                }
            ]

        create_body: dict[str, Any] = {
            "Image": self.image,
            "Env": env_list,
            "ExposedPorts": {f"{self.ollama_port}/tcp": {}},
            "HostConfig": host_config,
        }

        # Create the container
        result = self._docker_request(
            "POST",
            f"/containers/create?name={container_name}",
            body=create_body,
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected response from container create: {result}")
        container_id = result.get("Id", "")
        if not container_id:
            raise RuntimeError(f"Container creation failed: {result}")

        # Start the container
        self._docker_request("POST", f"/containers/{container_id}/start")

        # Inspect to get the assigned port
        inspect_data = self._docker_request("GET", f"/containers/{container_id}/json")
        if not isinstance(inspect_data, dict):
            inspect_data = {}

        assigned_port = host_port
        if not assigned_port:
            port_bindings = (
                inspect_data.get("NetworkSettings", {})
                .get("Ports", {})
                .get(f"{self.ollama_port}/tcp", [])
            )
            if port_bindings:
                assigned_port = int(port_bindings[0].get("HostPort", self.ollama_port))
            else:
                assigned_port = self.ollama_port

        endpoint = f"http://localhost:{assigned_port}"

        node = ComputeNode(
            node_id=f"docker-{container_id[:12]}",
            backend=ComputeBackend.DOCKER,
            endpoint=endpoint,
            model=model or "",
            max_concurrent=4,
            metadata={
                "container_id": container_id,
                "container_name": container_name,
                "image": self.image,
                "gpu_enabled": self.gpu_enabled,
            },
        )

        logger.info(
            "Started Ollama container %s at %s (gpu=%s)",
            container_name,
            endpoint,
            self.gpu_enabled,
        )
        return node

    async def stop_node(self, node_id: str) -> None:
        """Stop an Ollama container.

        Sends a stop signal to the container and removes it.

        Args:
            node_id: The ``node_id`` from the :class:`ComputeNode` (e.g.
                     ``docker-abc123``).
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._stop_node_sync, node_id)

    def _stop_node_sync(self, node_id: str) -> None:
        """Synchronous implementation of container stop + remove."""
        # node_id format: docker-<container_short_id>
        container_id = node_id.replace("docker-", "", 1)

        try:
            self._docker_request("POST", f"/containers/{container_id}/stop?t=10")
        except RuntimeError as exc:
            logger.warning("Error stopping container %s: %s", container_id, exc)

        try:
            self._docker_request("DELETE", f"/containers/{container_id}?force=true")
        except RuntimeError as exc:
            logger.warning("Error removing container %s: %s", container_id, exc)

        logger.info("Stopped and removed container %s", container_id)

    async def pull_model(self, node: ComputeNode, model: str) -> bool:
        """Pull a model on a running Ollama container.

        Sends a POST to the Ollama ``/api/pull`` endpoint inside the
        container.

        Args:
            node:  The node to pull the model on.
            model: Model name (e.g. ``"llama3"``).

        Returns:
            ``True`` if the pull completed successfully.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._pull_model_sync, node.endpoint, model
        )

    @staticmethod
    def _pull_model_sync(endpoint: str, model: str) -> bool:
        """Synchronous model pull."""
        url = f"{endpoint.rstrip('/')}/api/pull"
        payload = json.dumps({"name": model, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)  # Models can be large
            return resp.status == 200
        except Exception as exc:
            logger.warning("Failed to pull model %s: %s", model, exc)
            return False
