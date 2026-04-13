"""ServerApp entrypoint for Phase 2 SUMO-based MAVFL runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .data import load_global_eval_data
from .helpers import DelayParams
from .mobility_sumo import SUMOMobilityTrace
from .models import build_model
from .strategies import CBSStrategy, RBSStrategy, RandomStrategy, UCBStrategy
from .task import evaluate_model, get_device, set_global_seed

app = ServerApp()


def _model_size_bits(model: torch.nn.Module) -> float:
    total_bytes = 0
    for param in model.state_dict().values():
        total_bytes += param.numel() * param.element_size()
    return float(total_bytes * 8)


def _sample_bits(task_name: str) -> float:
    if task_name in {"cifar10_resnet18", "gtsrb_lenet"}:
        return float(32 * 32 * 3 * 8)
    raise ValueError(f"Unsupported task '{task_name}' for sample bits")


def _make_strategy(context: Context, model_size_bits: float, data_bits_per_client: float):
    strategy_name = str(context.run_config["strategy"])
    common_kwargs = {
        "fraction_train": float(context.run_config["fraction-train"]),
        "fraction_evaluate": float(context.run_config["fraction-evaluate"]),
        "min_train_nodes": int(context.run_config["min-train-nodes"]),
        "min_evaluate_nodes": int(context.run_config["min-evaluate-nodes"]),
        "min_available_nodes": int(context.run_config["min-available-nodes"]),
    }

    if strategy_name == "fedavg":
        return FedAvg(**common_kwargs)

    delay_params = DelayParams(
        bandwidth_hz=float(context.run_config["bandwidth-hz"]),
        tx_power_dbm=float(context.run_config["tx-power-db"]),
        noise_power_dbm=float(context.run_config["noise-power-db"]),
        bs_antenna_gain_db=float(context.run_config["bs-antenna-gain-db"]),
        path_loss_a=float(context.run_config["path-loss-a"]),
        path_loss_b=float(context.run_config["path-loss-b"]),
        gpu_frequency_ghz=float(context.run_config["gpu-frequency-ghz"]),
        gpu_cycles_per_bit=float(context.run_config["gpu-cycles-per-bit"]),
        compute_normalization=float(context.run_config["compute-normalization"]),
    )

    # Phase 2: SUMO trace-based mobility
    sumo_seed = int(context.run_config["sumo-seed"])
    trace_dir = Path(f"traces/seed_{sumo_seed}")
    mobility = SUMOMobilityTrace(
        trace_csv=str(trace_dir / "dublin_morning.csv"),
        vehicle_map_json=str(trace_dir / "vehicle_id_map.json"),
        coverage_radius_m=float(context.run_config["coverage-radius-m"]),
        bs_height_m=float(context.run_config["bs-height-m"]),
        num_zones=int(context.run_config["num-zones"]),
    )

    selection_size = int(context.run_config["num-selected"])
    base_kwargs = {
        **common_kwargs,
        "mobility": mobility,
        "delay_params": delay_params,
        "selection_size": selection_size,
        "alpha": float(context.run_config["mavfl-alpha"]),
        "model_size_bits": model_size_bits,
        "data_bits_per_client": data_bits_per_client,
        "seed": int(context.run_config["seed"]),
    }

    if strategy_name == "random":
        return RandomStrategy(**base_kwargs)
    if strategy_name == "cbs":
        return CBSStrategy(**base_kwargs)
    if strategy_name == "rbs":
        return RBSStrategy(**base_kwargs)
    if strategy_name == "ucb":
        return UCBStrategy(
            **base_kwargs,
            ucb_discount=float(context.run_config["ucb-discount"]),
            ucb_exploration=float(context.run_config["ucb-exploration"]),
        )

    raise ValueError("strategy must be 'fedavg', 'random', 'cbs', 'rbs', or 'ucb'")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run federated training with baseline or MAVFL strategy."""
    task_name = str(context.run_config["task"])
    batch_size = int(context.run_config["batch-size"])
    _ufd = context.run_config["use-fake-data"]
    use_fake_data = _ufd if isinstance(_ufd, bool) else str(_ufd).lower() == "true"
    train_samples_per_client = int(context.run_config["train-samples-per-client"])
    eval_samples_per_client = int(context.run_config["eval-samples-per-client"])

    set_global_seed(int(context.run_config["seed"]))
    model = build_model(task_name)
    arrays = ArrayRecord(model.state_dict())
    model_size_bits = _model_size_bits(model)
    data_bits_per_client = _sample_bits(task_name) * float(
        context.run_config["train-samples-per-client"]
    )

    eval_loader = load_global_eval_data(
        task_name=task_name,
        batch_size=batch_size,
        use_fake_data=use_fake_data,
        train_samples_per_client=train_samples_per_client,
        eval_samples_per_client=eval_samples_per_client,
    )

    def global_evaluate(server_round: int, global_arrays: ArrayRecord) -> MetricRecord:
        eval_model = build_model(task_name)
        eval_model.load_state_dict(global_arrays.to_torch_state_dict())
        device = get_device()
        eval_model.to(device)
        loss, acc = evaluate_model(eval_model, eval_loader, device)
        return MetricRecord({"server-round": float(server_round), "loss": loss, "accuracy": acc})

    strategy = _make_strategy(context, model_size_bits, data_bits_per_client)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=int(context.run_config["num-server-rounds"]),
        train_config=ConfigRecord(
            {
                "lr": float(context.run_config["learning-rate"]),
                "local-epochs": int(context.run_config["local-epochs"]),
                "task": task_name,
            }
        ),
        evaluate_config=ConfigRecord({"task": task_name}),
        evaluate_fn=global_evaluate,
    )

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = artifacts_dir / "final_model.pt"
    torch.save(result.arrays.to_torch_state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    metrics_path = artifacts_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "success-ratio",
                "round-delay-s",
                "utility",
                "dropout-count",
                "server-loss",
                "server-accuracy",
            ],
        )
        writer.writeheader()
        num_rounds = int(context.run_config["num-server-rounds"])
        for rnd in range(1, num_rounds + 1):
            train_mr = result.train_metrics_clientapp.get(rnd)
            server_mr = result.evaluate_metrics_serverapp.get(rnd)
            writer.writerow(
                {
                    "round": rnd,
                    "success-ratio": float(train_mr["mavfl-success-ratio"]) if train_mr and "mavfl-success-ratio" in train_mr else "",
                    "round-delay-s": float(train_mr["mavfl-round-delay-s"]) if train_mr and "mavfl-round-delay-s" in train_mr else "",
                    "utility": float(train_mr["mavfl-utility"]) if train_mr and "mavfl-utility" in train_mr else "",
                    "dropout-count": float(train_mr["mavfl-dropout-count"]) if train_mr and "mavfl-dropout-count" in train_mr else "",
                    "server-loss": float(server_mr["loss"]) if server_mr and "loss" in server_mr else "",
                    "server-accuracy": float(server_mr["accuracy"]) if server_mr and "accuracy" in server_mr else "",
                }
            )
    print(f"Saved metrics to {metrics_path}")

    if hasattr(strategy, "audit_trail"):
        audit_path = artifacts_dir / "ucb_audit_trail.json"
        with open(audit_path, "w") as f:
            json.dump(
                [record.__dict__ for record in getattr(strategy, "audit_trail", [])],
                f,
                indent=2,
            )
        print(f"Saved UCB audit trail to {audit_path}")
