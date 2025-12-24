# Ray Cluster Setup for Earthlink

This directory contains configuration for running Earthlink agents on a distributed Ray cluster.

## Files

- **`docker-compose.ray.yml`** - Docker Compose extension for Ray head node + workers
- **`ray-cluster.sh`** - Bash script to start Ray cluster manually
- **`ray-autoscaler.yaml`** - Ray autoscaler configuration

## Usage

### Option 1: Docker Compose (Recommended)

Start the full stack including Ray cluster:

```bash
docker-compose -f docker-compose.yml -f docker-compose.ray.yml up -d
```

This starts:
- Ray head node (dashboard on port 8265)
- All backend services (PostgreSQL, Redis, ChromaDB, Ollama, API)

### Option 2: Manual Ray Cluster

Start head node:
```bash
ray start --head --port=6380 --dashboard-host=0.0.0.0 --dashboard-port=8265
```

Start worker nodes (on separate machines):
```bash
ray start --address='<head-node-ip>:6380'
```

### Option 3: Ray Autoscaler

Use the autoscaler config to manage cluster scaling:
```bash
ray up ray-autoscaler.yaml
```

## API Usage

### Check Cluster Status

```bash
curl http://localhost:8000/api/v1/simulation/cluster
```

### Create Placement Group

Create a placement group for distributing agents:

```bash
curl -X POST http://localhost:8000/api/v1/simulation/cluster/placement-groups \
  -H "Content-Type: application/json" \
  -d '{
    "name": "agent-group-1",
    "num_agents": 100,
    "strategy": "SPREAD"
  }'
```

Strategies:
- **`SPREAD`** - Distribute agents across nodes (load balancing)
- **`PACK`** - Pack agents on same node (communication efficiency)
- **`STRICT_SPREAD`** - Strictly one bundle per node

### Spawn Agent with Placement Group

```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "A1",
    "placement_group": "agent-group-1"
  }'
```

## Scaling to 10,000 Agents

For large-scale deployments:

1. **Create placement groups** for agent batches:
   ```python
   # Create 10 placement groups, 1000 agents each
   for i in range(10):
       await simulation.create_agent_placement_group(
           name=f"pg-{i}",
           num_agents=1000,
           strategy="SPREAD"
       )
   ```

2. **Spawn agents** into placement groups:
   ```python
   for i in range(10000):
       pg_index = i // 1000
       await simulation.spawn_agent(
           name=f"A{i+1}",
           placement_group=f"pg-{pg_index}"
       )
   ```

3. **Monitor cluster** resources:
   ```bash
   curl http://localhost:8000/api/v1/simulation/cluster
   ```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Ray Head Node                          │
│  - Cluster coordination                                     │
│  - Resource management                                      │
│  - Dashboard (port 8265)                                    │
│  - Autoscaler                                               │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
┌─────────▼──────┐ ┌───────▼──────┐ ┌──────▼───────┐
│  Worker Node 1 │ │ Worker Node 2│ │ Worker Node N│
│  - Agents      │ │ - Agents     │ │ - Agents     │
│  - Tasks       │ │ - Tasks      │ │ - Tasks      │
└────────────────┘ └──────────────┘ └──────────────┘
```

## Monitoring

- **Ray Dashboard**: http://localhost:8265
- **Cluster Stats API**: http://localhost:8000/api/v1/simulation/cluster
- **Metrics Export**: http://localhost:8080/metrics

## Notes

- Each placement group pre-allocates resources for agents
- Autoscaler adds/removes workers based on resource demand
- For GPU-enabled agents, ensure workers have GPU access
- Use SPREAD strategy for load balancing, PACK for co-located communication
