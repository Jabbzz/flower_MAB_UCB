"""ClientApp for Phase 2 federated workloads.

Simplified from Phase 1: no mobility rollout, no client-side dropout check.
Dropout is handled server-side via dwell_remaining in strategy_base.py.
The client always trains and returns SUCCESS.
"""

from __future__ import annotations

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .data import load_partition_data
from .models import build_model
from .task import evaluate_model, get_device, set_global_seed, train_model

app = ClientApp()


def _resolve_client_partition(context: Context, cfg: ConfigRecord) -> tuple[int, int]:
    """Resolve shard-based partition for data loading.

    Phase 2: shard-id is sent by the server in the config record.
    """
    cid = int(cfg.get("shard-id", context.node_id % 200))
    num_clients = int(context.run_config["num-clients"])
    return cid, num_clients


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train selected client and return updated weights + metrics."""
    cfg = msg.content["config"]
    task_name = str(cfg["task"])

    model = build_model(task_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    cid, num_clients = _resolve_client_partition(context, cfg)
    seed = int(context.run_config["seed"])
    set_global_seed(seed + cid)
    batch_size = int(context.run_config["batch-size"])
    _ufd = context.run_config["use-fake-data"]
    use_fake_data = _ufd if isinstance(_ufd, bool) else str(_ufd).lower() == "true"
    train_samples_per_client = int(context.run_config["train-samples-per-client"])
    eval_samples_per_client = int(context.run_config["eval-samples-per-client"])

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

    # Phase 2: no mobility rollout, no dropout check — always return SUCCESS
    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(
                {
                    "num-examples": len(train_loader.dataset),
                    "train-loss": train_loss,
                    "train-accuracy": train_acc,
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

    cid, num_clients = _resolve_client_partition(context, cfg)
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
