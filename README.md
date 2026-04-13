# flower_MAB_UCB

This is the supplementary code for my bachelor dissertation: **Driven by Bandits: Client Selection in Vehicular Federated Learning Under Realistic Urban Mobility**

## Project Structure

```
├── phase_1/           # IDM-based synthetic mobility simulations
│   ├── flower/        # Flower app (client, server, strategies, mobility)
│   ├── configs/       # Run-config overrides (TOML files)
│   └── pyproject.toml # Simulation parameters
├── phase_2/           # SUMO trace-driven mobility simulations
│   ├── flower/        # Flower app (client, server, strategies, SUMO mobility)
│   ├── traces/        # SUMO trace data
│   └── pyproject.toml # Simulation parameters
```

The strategies tested in this work:

- **UCB** -- Upper Confidence Bound
- **CBS** -- Communication-Based Selection
- **RBS** -- Remainingtime-Based Selection
- **Random** -- random selection

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** -- fast Python package manager
- **Flower configuration** -- the simulation federation is configured in `~/.flwr/config.toml`. You may need to adjust `num-supernodes` and `client-resources` to match your machine:

```toml
[superlink]
default = "local-sim"

[superlink.local-sim]
options.num-supernodes = 200
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0
```

Set `num-gpus` to a fraction (e.g. `0.1`) if you have a GPU available.

## Installation & Running

### Phase 1 (Synthetic IDM Mobility)

```bash
cd phase_1
uv sync                # install dependencies into a virtual environment
uv pip install .       # install the project package
uv run flwr run .             # run the simulation
```

### Phase 2 (SUMO Trace-Driven Mobility)

```bash
cd phase_2
uv sync
uv pip install .
uv run flwr run .
```

### Overriding Run Config

You can override any config parameter at the command line:

```bash
uv run flwr run . --run-config "num-server-rounds=50 strategy=ucb num-clients=300"
```

Or point to a config file (Phase 1):

```bash
uv run flwr run . --run-config configs/gtsrb_mavfl.toml
```

You can debug simulations with the `--stream` flag to receive more information:
```bash
uv run flwr run . --stream
```


