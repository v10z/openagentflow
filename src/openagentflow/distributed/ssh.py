"""SSH backend for distributed Ollama inference on remote machines.

Manages Ollama instances on bare-metal servers or VMs reachable via SSH.
Uses ``subprocess`` to invoke the ``ssh`` binary (available on all major
platforms including Windows 10+).

No ``paramiko``, ``fabric``, or other SSH library is required.

Useful for:
    - Research clusters without Kubernetes
    - Bare-metal GPU servers
    - Cloud VMs without container orchestration
    - On-premises machines with heterogeneous GPUs

Example::

    from openagentflow.distributed.ssh import SSHBackend

    hosts = [
        {"host": "gpu1.local", "port": 22, "user": "admin", "ollama_port": 11434},
        {"host": "gpu2.local", "port": 22, "user": "admin", "ollama_port": 11434},
    ]

    ssh = SSHBackend(hosts=hosts)
    nodes = await ssh.discover_nodes()
    for node in nodes:
        ok = await ssh.health_check(node)
        print(f"{node.node_id}: {'OK' if ok else 'DOWN'}")

    # Start Ollama on a host where it's not running
    await ssh.start_ollama("gpu2.local")
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import urllib.error
import urllib.request
from typing import Any

from openagentflow.distributed.base import (
    ComputeBackend,
    ComputeBackendInterface,
    ComputeNode,
)

logger = logging.getLogger(__name__)


def _ssh_host_id(host_cfg: dict[str, Any]) -> str:
    """Derive a stable node-id string from a host config dict."""
    host = host_cfg.get("host", "unknown")
    port = host_cfg.get("ollama_port", 11434)
    return f"ssh-{host}-{port}"


class SSHBackend(ComputeBackendInterface):
    """Manage Ollama instances on remote machines via SSH.

    Each host is described by a dict with the following keys:

    ========== ======== ==============================================
    Key        Default  Description
    ========== ======== ==============================================
    host       (req.)   Hostname or IP address
    port       22       SSH port
    user       (curr.)  SSH user name (defaults to current OS user)
    ollama_port 11434   Port Ollama listens on on the remote host
    key_file   None     Path to SSH private key (optional)
    ========== ======== ==============================================

    SSH connections use ``-o StrictHostKeyChecking=no`` and
    ``-o ConnectTimeout=10`` for robustness.  For production deployments
    it is recommended to set up SSH keys and ``known_hosts`` properly.

    Args:
        hosts:          List of host configuration dicts (see table above).
        ssh_binary:     Path to the ``ssh`` executable (default ``"ssh"``).
        connect_timeout: SSH connection timeout in seconds.
    """

    def __init__(
        self,
        hosts: list[dict[str, Any]],
        ssh_binary: str = "ssh",
        connect_timeout: int = 10,
    ) -> None:
        self.hosts = hosts
        self.ssh_binary = ssh_binary
        self.connect_timeout = connect_timeout

    # ------------------------------------------------------------------
    # SSH helpers
    # ------------------------------------------------------------------

    def _build_ssh_command(
        self, host_cfg: dict[str, Any], remote_command: str
    ) -> list[str]:
        """Build the ``ssh`` command-line for a remote host.

        Args:
            host_cfg:       Host configuration dict.
            remote_command: The command to execute on the remote host.

        Returns:
            List of strings suitable for ``subprocess.run``.
        """
        host = host_cfg["host"]
        port = str(host_cfg.get("port", 22))
        user = host_cfg.get("user", "")

        cmd: list[str] = [self.ssh_binary]
        cmd += ["-o", "StrictHostKeyChecking=no"]
        cmd += ["-o", f"ConnectTimeout={self.connect_timeout}"]
        cmd += ["-o", "BatchMode=yes"]  # Never prompt for password
        cmd += ["-p", port]

        key_file = host_cfg.get("key_file")
        if key_file:
            cmd += ["-i", key_file]

        target = f"{user}@{host}" if user else host
        cmd.append(target)
        cmd.append(remote_command)

        return cmd

    def _run_ssh(
        self, host_cfg: dict[str, Any], remote_command: str, timeout: int = 30
    ) -> tuple[bool, str]:
        """Run a command on a remote host via SSH.

        Args:
            host_cfg:       Host configuration dict.
            remote_command: Shell command to execute remotely.
            timeout:        Maximum time to wait for the command (seconds).

        Returns:
            Tuple of (success: bool, output: str).
        """
        cmd = self._build_ssh_command(host_cfg, remote_command)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout.strip()
            if result.returncode == 0:
                return True, output
            else:
                err = result.stderr.strip()
                logger.debug(
                    "SSH command failed on %s (rc=%d): %s",
                    host_cfg["host"],
                    result.returncode,
                    err,
                )
                return False, err
        except subprocess.TimeoutExpired:
            logger.warning(
                "SSH command timed out on %s after %ds",
                host_cfg["host"],
                timeout,
            )
            return False, "timeout"
        except FileNotFoundError:
            logger.error(
                "SSH binary not found at '%s'. Ensure OpenSSH is installed.",
                self.ssh_binary,
            )
            return False, "ssh binary not found"
        except Exception as exc:
            logger.warning("SSH error on %s: %s", host_cfg["host"], exc)
            return False, str(exc)

    def _http_health_check(self, endpoint: str, timeout: int = 5) -> bool:
        """Check Ollama health by hitting ``/api/tags`` over HTTP."""
        url = f"{endpoint.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            return resp.status == 200
        except Exception:
            return False

    def _http_get_models(self, endpoint: str, timeout: int = 5) -> str:
        """Get the first loaded model name from ``/api/tags``."""
        url = f"{endpoint.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            body = json.loads(resp.read().decode("utf-8"))
            models = body.get("models", [])
            if models:
                return models[0].get("name", "")
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # ComputeBackendInterface implementation
    # ------------------------------------------------------------------

    async def discover_nodes(self) -> list[ComputeNode]:
        """Check which configured hosts have Ollama running.

        For each host, first attempts a direct HTTP health check to the
        Ollama port.  If that fails (e.g. the host is behind a firewall),
        falls back to SSHing in and checking if the Ollama process is
        running.

        Returns:
            List of :class:`ComputeNode` instances for reachable Ollama servers.
        """
        loop = asyncio.get_running_loop()
        # Run all checks in parallel via the thread executor
        tasks = [
            loop.run_in_executor(None, self._discover_single_node, host_cfg)
            for host_cfg in self.hosts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        nodes: list[ComputeNode] = []
        for result in results:
            if isinstance(result, ComputeNode):
                nodes.append(result)
            elif isinstance(result, Exception):
                logger.debug("Node discovery failed: %s", result)

        logger.info("Discovered %d Ollama hosts via SSH backend", len(nodes))
        return nodes

    def _discover_single_node(self, host_cfg: dict[str, Any]) -> ComputeNode | None:
        """Discover a single host (synchronous)."""
        host = host_cfg["host"]
        ollama_port = host_cfg.get("ollama_port", 11434)
        endpoint = f"http://{host}:{ollama_port}"

        # Try direct HTTP first
        if self._http_health_check(endpoint):
            model = self._http_get_models(endpoint)
            return ComputeNode(
                node_id=_ssh_host_id(host_cfg),
                backend=ComputeBackend.SSH,
                endpoint=endpoint,
                model=model,
                max_concurrent=4,
                metadata={
                    "host": host,
                    "ssh_port": host_cfg.get("port", 22),
                    "user": host_cfg.get("user", ""),
                    "discovery_method": "http",
                },
            )

        # Fall back to SSH check
        success, output = self._run_ssh(
            host_cfg,
            "pgrep -x ollama > /dev/null 2>&1 && echo RUNNING || echo STOPPED",
        )
        if success and "RUNNING" in output:
            return ComputeNode(
                node_id=_ssh_host_id(host_cfg),
                backend=ComputeBackend.SSH,
                endpoint=endpoint,
                max_concurrent=4,
                metadata={
                    "host": host,
                    "ssh_port": host_cfg.get("port", 22),
                    "user": host_cfg.get("user", ""),
                    "discovery_method": "ssh",
                },
            )

        return None

    async def health_check(self, node: ComputeNode) -> bool:
        """Check if an Ollama instance on a remote host is healthy.

        Attempts an HTTP health check first.  If the endpoint is not directly
        reachable (e.g. behind a firewall), SSHes into the host and runs
        ``curl`` locally.

        Args:
            node: The node to check.

        Returns:
            ``True`` if Ollama is running and responsive.
        """
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(None, self._health_check_sync, node)
        if ok:
            node.mark_healthy()
        else:
            node.mark_unhealthy()
        return ok

    def _health_check_sync(self, node: ComputeNode) -> bool:
        """Synchronous health check."""
        # Direct HTTP
        if self._http_health_check(node.endpoint):
            return True

        # SSH fallback: find the host config for this node
        host_cfg = self._find_host_config(node)
        if not host_cfg:
            return False

        ollama_port = host_cfg.get("ollama_port", 11434)
        success, output = self._run_ssh(
            host_cfg,
            f"curl -sf http://localhost:{ollama_port}/api/tags > /dev/null 2>&1"
            f" && echo OK || echo FAIL",
        )
        return success and "OK" in output

    def _find_host_config(self, node: ComputeNode) -> dict[str, Any] | None:
        """Find the host config dict for a given compute node."""
        target_host = node.metadata.get("host", "")
        for cfg in self.hosts:
            if cfg["host"] == target_host:
                return cfg
        return None

    # ------------------------------------------------------------------
    # Ollama lifecycle management
    # ------------------------------------------------------------------

    async def start_ollama(self, host: str) -> bool:
        """Start Ollama on a remote host via SSH.

        Runs ``ollama serve`` in the background on the target host.  The
        command is daemonised with ``nohup`` so it survives SSH disconnection.

        Args:
            host: Hostname or IP matching one of the configured hosts.

        Returns:
            ``True`` if Ollama was started (or was already running).
        """
        host_cfg = self._find_host_by_name(host)
        if not host_cfg:
            logger.error("Host %s not found in SSH backend configuration", host)
            return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._start_ollama_sync, host_cfg)

    def _start_ollama_sync(self, host_cfg: dict[str, Any]) -> bool:
        """Synchronous implementation of start_ollama."""
        host = host_cfg["host"]

        # Check if already running
        success, output = self._run_ssh(
            host_cfg,
            "pgrep -x ollama > /dev/null 2>&1 && echo RUNNING || echo STOPPED",
        )
        if success and "RUNNING" in output:
            logger.info("Ollama is already running on %s", host)
            return True

        # Start Ollama in the background
        success, output = self._run_ssh(
            host_cfg,
            "nohup ollama serve > /tmp/ollama.log 2>&1 & echo STARTED",
            timeout=15,
        )
        if success and "STARTED" in output:
            logger.info("Started Ollama on %s", host)
            return True

        logger.warning("Failed to start Ollama on %s: %s", host, output)
        return False

    async def stop_ollama(self, host: str) -> bool:
        """Stop Ollama on a remote host via SSH.

        Sends ``SIGTERM`` to the Ollama process.

        Args:
            host: Hostname or IP matching one of the configured hosts.

        Returns:
            ``True`` if Ollama was stopped successfully.
        """
        host_cfg = self._find_host_by_name(host)
        if not host_cfg:
            logger.error("Host %s not found in SSH backend configuration", host)
            return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._stop_ollama_sync, host_cfg)

    def _stop_ollama_sync(self, host_cfg: dict[str, Any]) -> bool:
        """Synchronous implementation of stop_ollama."""
        success, output = self._run_ssh(
            host_cfg,
            "pkill -x ollama && echo STOPPED || echo NOT_RUNNING",
        )
        if success:
            logger.info("Stopped Ollama on %s", host_cfg["host"])
            return True
        return False

    async def get_gpu_info(self, host: str) -> dict[str, Any]:
        """Get GPU information from a remote host.

        Runs ``nvidia-smi --query-gpu=...`` on the remote host.

        Args:
            host: Hostname or IP.

        Returns:
            Dict with ``gpu_count``, ``gpu_names``, ``total_memory_mb``,
            and ``free_memory_mb``.
        """
        host_cfg = self._find_host_by_name(host)
        if not host_cfg:
            return {"gpu_count": 0, "gpu_names": [], "total_memory_mb": 0, "free_memory_mb": 0}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_gpu_info_sync, host_cfg)

    def _get_gpu_info_sync(self, host_cfg: dict[str, Any]) -> dict[str, Any]:
        """Synchronous GPU info retrieval."""
        success, output = self._run_ssh(
            host_cfg,
            "nvidia-smi --query-gpu=name,memory.total,memory.free "
            "--format=csv,noheader,nounits 2>/dev/null || echo NO_GPU",
        )

        result: dict[str, Any] = {
            "gpu_count": 0,
            "gpu_names": [],
            "total_memory_mb": 0,
            "free_memory_mb": 0,
        }

        if not success or "NO_GPU" in output:
            return result

        lines = [line.strip() for line in output.splitlines() if line.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                result["gpu_count"] += 1
                result["gpu_names"].append(parts[0])
                try:
                    result["total_memory_mb"] += int(float(parts[1]))
                    result["free_memory_mb"] += int(float(parts[2]))
                except (ValueError, IndexError):
                    pass

        return result

    async def pull_model(self, host: str, model: str) -> bool:
        """Pull a model on a remote Ollama instance.

        Args:
            host:  Hostname or IP.
            model: Model name (e.g. ``"llama3"``).

        Returns:
            ``True`` if the pull succeeded.
        """
        host_cfg = self._find_host_by_name(host)
        if not host_cfg:
            return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._pull_model_sync, host_cfg, model)

    def _pull_model_sync(self, host_cfg: dict[str, Any], model: str) -> bool:
        """Synchronous model pull."""
        success, output = self._run_ssh(
            host_cfg,
            f"ollama pull {model} 2>&1",
            timeout=600,  # Large models take time to download
        )
        if success:
            logger.info("Pulled model %s on %s", model, host_cfg["host"])
            return True
        logger.warning("Failed to pull model %s on %s: %s", model, host_cfg["host"], output)
        return False

    def _find_host_by_name(self, host: str) -> dict[str, Any] | None:
        """Find a host config dict by hostname / IP."""
        for cfg in self.hosts:
            if cfg["host"] == host:
                return cfg
        return None
