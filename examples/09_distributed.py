"""
OpenAgentFlow Example 09: Distributed Compute

Demonstrates:
- Creating compute clusters
- Adding nodes with different backends
- Node selection and availability
"""
from openagentflow.distributed import ComputeCluster, ComputeNode, ComputeBackend


def main():
    print("=" * 60)
    print("OpenAgentFlow Example 09: Distributed Compute")
    print("=" * 60)

    # Create a cluster
    cluster = ComputeCluster(name="inference-cluster")
    print(f"\nCluster: {cluster.name}")

    # Add nodes with different backends
    # Note: ComputeNode uses 'model' for the loaded model and 'metadata' for
    # arbitrary key-value labels (type, location, GPU info, etc.).
    nodes = [
        ComputeNode(
            node_id="ollama-local",
            backend=ComputeBackend.HTTP,
            endpoint="http://localhost:11434",
            model="llama3",
            metadata={"type": "cpu", "location": "local"},
        ),
        ComputeNode(
            node_id="gpu-server-1",
            backend=ComputeBackend.HTTP,
            endpoint="http://gpu1.internal:11434",
            model="llama3",
            gpu_memory_mb=40960,
            metadata={"type": "gpu", "gpu_model": "a100", "location": "datacenter"},
        ),
        ComputeNode(
            node_id="k8s-pod-pool",
            backend=ComputeBackend.KUBERNETES,
            endpoint="k8s://ollama-deployment",
            model="llama3",
            gpu_memory_mb=40960,
            metadata={"type": "gpu", "gpu_model": "a100", "location": "cloud"},
        ),
        ComputeNode(
            node_id="docker-worker",
            backend=ComputeBackend.DOCKER,
            endpoint="docker://ollama-container",
            model="mistral",
            metadata={"type": "cpu", "location": "local"},
        ),
        ComputeNode(
            node_id="ssh-remote",
            backend=ComputeBackend.SSH,
            endpoint="ssh://user@remote-host:22",
            model="llama3",
            gpu_memory_mb=81920,
            metadata={"type": "gpu", "gpu_model": "h100", "location": "remote"},
        ),
    ]

    for node in nodes:
        cluster.add_node(node)
        print(f"  Added: {node.node_id} ({node.backend.value}) -> {node.endpoint}")

    # List all nodes
    print(f"\nTotal nodes: {len(cluster.nodes)}")

    # Get available nodes
    available = cluster.get_available_nodes()
    print(f"Available nodes: {len(available)}")
    for node in available:
        print(f"  {node.node_id}: {node.backend.value} model={node.model} [{', '.join(f'{k}={v}' for k, v in node.metadata.items())}]")

    # Select by metadata
    print("\n--- Node Selection ---")
    gpu_nodes = [n for n in cluster.nodes if n.metadata.get("type") == "gpu"]
    print(f"GPU nodes: {[n.node_id for n in gpu_nodes]}")

    a100_nodes = [n for n in cluster.nodes if n.metadata.get("gpu_model") == "a100"]
    print(f"A100 nodes: {[n.node_id for n in a100_nodes]}")

    local_nodes = [n for n in cluster.nodes if n.metadata.get("location") == "local"]
    print(f"Local nodes: {[n.node_id for n in local_nodes]}")

    print("\n" + "=" * 60)
    print("Distributed compute demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
