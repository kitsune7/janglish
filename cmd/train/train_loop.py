from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

import torch
from data_collator import FrameLabelCollator
from dataset import training_data, validation_data
from dotenv import load_dotenv
from model_setup import feature_extractor, model
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional
from torch.utils.data import DataLoader
from transformers import get_scheduler

import wandb

load_dotenv()
os.environ.setdefault("WANDB_PROJECT", "janglish")

if len(training_data) == 0:
    raise RuntimeError("No training examples found. Check data/data-pairs.csv and matching .flac/.json files.")


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path = Path("./models/lid-wav2vec2")
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # effective batch 8; tune to your VRAM
    learning_rate: float = 3e-5  # standard for wav2vec2 fine-tuning
    warmup_steps: int = 4
    num_train_epochs: int = 20
    lr_scheduler_type: str = "linear"
    logging_steps: int = 5
    run_name: str = "lid-wav2vec2"
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False


def main() -> None:
    config = TrainingConfig()
    device = resolve_device()
    collator = FrameLabelCollator(feature_extractor)
    train_loader = DataLoader(
        training_data,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer_steps_per_epoch = ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_optimizer_steps = config.num_train_epochs * optimizer_steps_per_epoch
    lr_scheduler = get_scheduler(
        config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        name=config.run_name,
        config=wandb_config(config, device, total_optimizer_steps),
    )
    try:
        train(model, train_loader, validation_loader, optimizer, lr_scheduler, config, device)
    finally:
        if wandb.run is not None:
            wandb.finish()


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def wandb_config(config: TrainingConfig, device: torch.device, total_optimizer_steps: int) -> dict[str, object]:
    values = asdict(config)
    values["output_dir"] = str(config.output_dir)
    values["device"] = str(device)
    values["total_optimizer_steps"] = total_optimizer_steps
    return values


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = float("-inf")
    global_step = 0

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, config.num_train_epochs + 1):
        model.train()
        recent_loss = 0.0
        recent_batches = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            with autocast_context(device):
                loss, _ = compute_loss(model, batch)
                scaled_loss = loss / config.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            recent_loss += loss.item()
            recent_batches += 1

            should_step = (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if not should_step:
                continue

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % config.logging_steps == 0:
                average_loss = recent_loss / recent_batches
                wandb.log(
                    {
                        "train/loss": average_loss,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=global_step,
                )
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={average_loss:.4f} lr={lr_scheduler.get_last_lr()[0]:.2e}"
                )
                recent_loss = 0.0
                recent_batches = 0

        metrics = evaluate(model, validation_loader, device)
        wandb.log(prefix_metrics(metrics, "validation") | {"epoch": epoch}, step=global_step)
        print(
            f"epoch={epoch} validation "
            f"loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            save_best_checkpoint(config.output_dir, metrics, epoch, global_step)
            wandb.log({"best/f1_macro": best_f1, "best/epoch": epoch}, step=global_step)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"]
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = model(**inputs)
    logits = outputs.logits

    mask = labels.sum(-1) > 0
    targets = labels.argmax(-1)
    if mask.any():
        loss = functional.cross_entropy(logits[mask], targets[mask])
    else:
        loss = logits.sum() * 0

    return loss, logits


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float | list[float]]:
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    total_frames = 0

    for batch in data_loader:
        batch = move_to_device(batch, device)
        with autocast_context(device):
            loss, logits = compute_loss(model, batch)

        labels = batch["labels"]
        mask = labels.sum(-1) > 0
        if not mask.any():
            continue

        frame_count = mask.sum().item()
        total_loss += loss.item() * frame_count
        total_frames += frame_count
        all_predictions.append(logits.argmax(-1)[mask].cpu())
        all_labels.append(labels.argmax(-1)[mask].cpu())

    if not all_labels:
        return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_per_class": [0.0, 0.0]}

    predictions = torch.cat(all_predictions).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "loss": total_loss / total_frames,
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", labels=[0, 1], zero_division=0),
        "f1_per_class": f1_score(labels, predictions, average=None, labels=[0, 1], zero_division=0).tolist(),
    }


def prefix_metrics(metrics: dict[str, float | list[float]], prefix: str) -> dict[str, float]:
    prefixed = {
        f"{prefix}/loss": float(metrics["loss"]),
        f"{prefix}/accuracy": float(metrics["accuracy"]),
        f"{prefix}/f1_macro": float(metrics["f1_macro"]),
    }
    per_class = metrics["f1_per_class"]
    if isinstance(per_class, list) and len(per_class) == 2:
        prefixed[f"{prefix}/f1_e"] = per_class[0]
        prefixed[f"{prefix}/f1_j"] = per_class[1]
    return prefixed


def save_best_checkpoint(
    output_dir: Path,
    metrics: dict[str, float | list[float]],
    epoch: int,
    global_step: int,
) -> None:
    checkpoint_dir = output_dir / "checkpoint-best"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    feature_extractor.save_pretrained(checkpoint_dir)
    print(
        f"saved best checkpoint to {checkpoint_dir} "
        f"epoch={epoch} step={global_step} f1_macro={metrics['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
