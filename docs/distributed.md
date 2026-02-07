# Distributed Compute

## Overview

OpenAgentFlow supports distributing agent workloads across multiple nodes using different backend technologies. A `ComputeCluster` manages a pool of `ComputeNode` instances, each representing a remote execution target. The cluster handles node selection, health checking, and label-based routing.

## Backends

| Backend      | Use Case                              |
|--------------|---------------------------------------|
| `HTTP`       | Remote Ollama instances, REST APIs    |
| `DOCKER`     | Container-based execution             |
| `KUBERNETES` | K8s pod scheduling                    |
| `SSH`        | Remote machine access                 |

## Usage

```python
from openagentflow.distributed import ComputeCluster, ComputeNode, ComputeBackend

cluster = ComputeCluster(name="inference")

cluster.add_node(ComputeNode(
    node_id="gpu1",
    backend=ComputeBackend.HTTP,
    endpoint="http://gpu1:11434",
))

cluster.add_node(ComputeNode(
    node_id="gpu2",
    backend=ComputeBackend.KUBERNETES,
    endpoint="k8s://ollama-deployment",
    labels={"gpu": "a100"},
))

available = cluster.get_available_nodes()
node = cluster.select_node(labels={"gpu": "a100"})
```

## Node Configuration

Each `ComputeNode` accepts the following fields:

| Field       | Type             | Required | Description                                  |
|-------------|------------------|----------|----------------------------------------------|
| `node_id`   | `str`            | Yes      | Unique identifier for the node               |
| `backend`   | `ComputeBackend` | Yes      | Execution backend type                       |
| `endpoint`  | `str`            | Yes      | Connection string for the backend            |
| `labels`    | `dict`           | No       | Key-value pairs for routing and selection    |
| `max_concurrent` | `int`       | No       | Maximum concurrent tasks (default: 1)        |
| `timeout_s` | `float`          | No       | Request timeout in seconds (default: 300)    |
| `healthy`   | `bool`           | No       | Current health status (default: `True`)      |

### Endpoint Formats

| Backend      | Endpoint Format                          |
|--------------|------------------------------------------|
| `HTTP`       | `http://host:port` or `https://host:port` |
| `DOCKER`     | `docker://container-name`                |
| `KUBERNETES` | `k8s://deployment-name` or `k8s://namespace/deployment` |
| `SSH`        | `ssh://user@host:port`                   |

## Cluster Management

### Adding and Removing Nodes

```python
# Add a node
cluster.add_node(ComputeNode(
    node_id="cpu1",
    backend=ComputeBackend.SSH,
    endpoint="ssh://user@192.168.1.10:22",
    labels={"type": "cpu"},
))

# Remove a node by ID
cluster.remove_node("cpu1")
```

### Health Checks

```python
# Check all nodes
health = await cluster.health_check()
for node_id, status in health.items():
    print(f"{node_id}: {'healthy' if status else 'unhealthy'}")

# Unhealthy nodes are automatically excluded from selection
available = cluster.get_available_nodes()
```

### Node Selection Strategies

The `select_node` method picks a node from the available pool. You can filter by labels:

```python
# Select any available node
node = cluster.select_node()

# Select a node with specific labels
node = cluster.select_node(labels={"gpu": "a100"})

# Select a node matching multiple labels
node = cluster.select_node(labels={"gpu": "a100", "region": "us-east"})
```

When multiple nodes match, the cluster uses a least-loaded strategy based on each node's current task count relative to its `max_concurrent` setting.

### Listing Nodes

```python
for node in cluster.list_nodes():
    print(f"{node.node_id} ({node.backend.value}) - {node.endpoint}")
    if node.labels:
        print(f"  Labels: {node.labels}")
```
