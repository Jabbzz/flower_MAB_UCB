"""ClientApp for Phase 1 federated workloads."""

from __future__ import annotations

from flwr.app import ArrayRecord, ConfigRecord, Context, Error, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.constant import ErrorCode

from .data import load_partition_data
from .mobility import IDMRoadMobility
from .models import build_model
from .task import evaluate_model, get_device, set_global_seed, train_model

app = ClientApp()


def _resolve_client_partition(context: Context) -> tuple[int, int]:
    cid = int(context.node_config.get("partition-id", context.node_id % 10))
    num_clients = int(context.node_config.get("num-partitions", context.run_config["num-clients"]))
    return cid, num_clients


def _rollout_during_compute(cfg) -> tuple[float, float]:
    """Advance the client's mobility state over the modeled compute interval."""
    mobility = IDMRoadMobility(
        seed=0,
        road_length_m=float(cfg["mobility-road-length-m"]),
        num_zones=int(cfg["mobility-num-zones"]),
        bs_height_m=float(cfg.get("mobility-bs-height-m", 25.0)),
        desired_speed_kmh=float(cfg.get("mobility-desired-speed-kmh", 60.0)),
        max_accel_mps2=float(cfg["mobility-idm-max-accel-mps2"]),
        comfort_decel_mps2=float(cfg["mobility-idm-comfort-decel-mps2"]),
        min_gap_m=float(cfg["mobility-idm-min-gap-m"]),
        time_headway_s=float(cfg["mobility-idm-time-headway-s"]),
        accel_exponent=float(cfg["mobility-idm-accel-exponent"]),
        time_step_s=float(cfg["mobility-time-step-s"]),
        arrival_rate_hz=0.0,
        initial_active=0,
        channel_compute=None,
    )
    rolled = mobility.rollout_for_client(
        position_m=float(cfg["mobility-position-m"]),
        speed_mps=float(cfg["mobility-speed-mps"]),
        leader_gap_m=float(cfg["mobility-leader-gap-m"]),
        leader_speed_mps=float(cfg["mobility-leader-speed-mps"]),
        duration_s=float(cfg["mobility-compute-time-s"]),
    )
    return rolled.position_m, rolled.speed_mps


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train selected client and return updated weights + metrics."""
    cfg = msg.content["config"]
    task_name = str(cfg["task"])

    model = build_model(task_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    cid, num_clients = _resolve_client_partition(context)
    seed = int(context.run_config["seed"])
    set_global_seed(seed + cid)
    batch_size = int(context.run_config["batch-size"])
    _ufd = context.run_config["use-fake-data"]
    use_fake_data = _ufd if isinstance(_ufd, bool) else str(_ufd).lower() == "true"
    train_samples_per_client = int(context.run_config["train-samples-per-client"])
    eval_samples_per_client = int(context.run_config["eval-samples-per-client"])

    # _ is a throwaway variable received from the function.
    train_loader, _ = load_partition_data(
        task_name=task_name,
        cid=cid,
        num_clients=num_clients,
        batch_size=batch_size,
        use_fake_data=use_fake_data,
        train_samples_per_client=train_samples_per_client,
        eval_samples_per_client=eval_samples_per_client,
        seed=seed + cid,
    )

    train_loss, train_acc = train_model(
        model=model,
        train_loader=train_loader,
        local_epochs=int(cfg["local-epochs"]),
        lr=float(cfg["lr"]),
        device=device,
    )

    end_position_m, end_speed_mps = _rollout_during_compute(cfg)
    context.state["mobility-last-train"] = ConfigRecord(
        {
            "position-m": end_position_m,
            "speed-mps": end_speed_mps,
            "server-round": int(cfg["server-round"]),
        }
    )

    if end_position_m >= float(cfg["mobility-road-length-m"]):
        context.state["mobility-last-result"] = ConfigRecord({"dropped": True})
        return Message(
            Error(
                ErrorCode.UNKNOWN,
                (
                    "vehicle left coverage after local training "
                    f"(end_position_m={end_position_m:.3f})"
                ),
            ),
            reply_to=msg,
        )

    context.state["mobility-last-result"] = ConfigRecord({"dropped": False})

    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(
                {
                    "num-examples": len(train_loader.dataset),
                    "train-loss": train_loss,
                    "train-accuracy": train_acc,
                    "client-end-position-m": end_position_m,
                    "client-end-speed-mps": end_speed_mps,
                }
            ),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate client model and return metrics only."""
    cfg = msg.content["config"]
    task_name = str(cfg["task"])

    model = build_model(task_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    cid, num_clients = _resolve_client_partition(context)
    batch_size = int(context.run_config["batch-size"])
    _ufd = context.run_config["use-fake-data"]
    use_fake_data = _ufd if isinstance(_ufd, bool) else str(_ufd).lower() == "true"
    train_samples_per_client = int(context.run_config["train-samples-per-client"])
    eval_samples_per_client = int(context.run_config["eval-samples-per-client"])

    _, eval_loader = load_partition_data(
        task_name=task_name,
        cid=cid,
        num_clients=num_clients,
        batch_size=batch_size,
        use_fake_data=use_fake_data,
        train_samples_per_client=train_samples_per_client,
        eval_samples_per_client=eval_samples_per_client,
    )

    eval_loss, eval_acc = evaluate_model(model, eval_loader, device)
    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "num-examples": len(eval_loader.dataset),
                    "eval-loss": eval_loss,
                    "eval-accuracy": eval_acc,
                }
            )
        }
    )
    return Message(content=content, reply_to=msg)
