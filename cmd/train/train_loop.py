from __future__ import annotations

import faulthandler
import os
import random
import resource
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

import numpy as np
import torch
from data_collator import FrameLabelCollator
from dataset import load_datasets
from dotenv import load_dotenv
from model_setup import load_model_and_feature_extractor
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional
from torch.utils.data import DataLoader, Subset
from transformers import Wav2Vec2FeatureExtractor, get_scheduler

import wandb

load_dotenv()
os.environ.setdefault("WANDB_PROJECT", "janglish")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 13
    output_dir: Path = Path("./models/lid-wav2vec2")
    train_batch_size: int = 2
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # effective batch 8; tune to your VRAM
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    num_train_epochs: int = 3
    lr_scheduler_type: str = "linear"
    logging_steps: int = 5
    evaluation_logging_steps: int = 100
    quick_eval_steps: int = 100  # run a quick validation check every N optimizer steps (0 disables)
    quick_eval_examples: int = 200  # size of the fixed validation subset used for quick checks
    run_name: str = "lid-wav2vec2"
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False


def main() -> None:
    faulthandler.enable()  # dump Python tracebacks on hard crashes (SIGSEGV, SIGABRT, SIGBUS)
    config = TrainingConfig()
    seed_everything(config.seed)
    training_data, validation_data, _ = load_datasets(seed=config.seed)
    if len(training_data) == 0:
        raise RuntimeError("No training examples found. Check data/data-pairs.csv and matching .flac/.json files.")

    device = resolve_device()
    model, feature_extractor = load_model_and_feature_extractor()
    collator = FrameLabelCollator(feature_extractor)
    train_generator = seeded_torch_generator(config.seed)
    validation_generator = seeded_torch_generator(config.seed)
    train_loader = DataLoader(
        training_data,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
        generator=validation_generator,
        worker_init_fn=seed_worker,
    )
    quick_validation_loader = build_quick_validation_loader(validation_data, collator, config)

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
    configure_wandb_metrics()
    try:
        train(
            model,
            feature_extractor,
            train_loader,
            validation_loader,
            quick_validation_loader,
            optimizer,
            lr_scheduler,
            config,
            device,
        )
    finally:
        if wandb.run is not None:
            wandb.finish()


def build_quick_validation_loader(
    validation_data,
    collator: FrameLabelCollator,
    config: TrainingConfig,
) -> DataLoader | None:
    """A small fixed validation subset for cheap mid-epoch checks.

    The subset is chosen once with a seeded shuffle so quick metrics are comparable
    across steps and runs. Full validation at epoch end still selects the best model.
    """
    if config.quick_eval_steps <= 0 or config.quick_eval_examples <= 0 or len(validation_data) == 0:
        return None
    subset_size = min(config.quick_eval_examples, len(validation_data))
    indices = torch.randperm(len(validation_data), generator=seeded_torch_generator(config.seed))[:subset_size]
    return DataLoader(
        Subset(validation_data, indices.tolist()),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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


def configure_wandb_metrics() -> None:
    wandb.define_metric("epoch")
    wandb.define_metric("train/epoch")
    wandb.define_metric("train/*", step_metric="train/epoch")
    wandb.define_metric("validation/*", step_metric="epoch")
    wandb.define_metric("validation_quick/*", step_metric="train/global_step")
    wandb.define_metric("best/*", step_metric="epoch")
    wandb.define_metric("chosen/*", step_metric="epoch")


def train(
    model: torch.nn.Module,
    feature_extractor: Wav2Vec2FeatureExtractor,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    quick_validation_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_validation_loss = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float | list[float]] | None = None
    best_epoch = 0
    best_global_step = 0
    best_train_loss = 0.0
    global_step = 0

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, config.num_train_epochs + 1):
        model.train()
        recent_loss = 0.0
        recent_batches = 0
        epoch_loss = 0.0
        epoch_batches = 0

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
            epoch_loss += loss.item()
            epoch_batches += 1

            should_step = batch_index % config.gradient_accumulation_steps == 0 or batch_index == len(train_loader)
            if not should_step:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % config.logging_steps == 0:
                average_loss = recent_loss / recent_batches
                epoch_progress = epoch - 1 + batch_index / len(train_loader)
                wandb.log(
                    {
                        "train/loss": average_loss,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch_progress,
                        "train/global_step": global_step,
                        "epoch": epoch,
                    },
                    step=global_step,
                )
                print(
                    f"epoch={epoch} step={global_step} loss={average_loss:.4f} lr={lr_scheduler.get_last_lr()[0]:.2e}"
                )
                recent_loss = 0.0
                recent_batches = 0

            if quick_validation_loader is not None and global_step % config.quick_eval_steps == 0:
                quick_metrics = evaluate(model, quick_validation_loader, device, logging_steps=0, phase="quick")
                model.train()
                wandb.log(
                    prefix_metrics(quick_metrics, "validation_quick") | {"train/global_step": global_step},
                    step=global_step,
                )
                print(
                    f"epoch={epoch} step={global_step} quick_validation_loss={quick_metrics['loss']:.4f} "
                    f"accuracy={quick_metrics['accuracy']:.4f} f1_macro={quick_metrics['f1_macro']:.4f} "
                    f"{memory_summary(device)}",
                    flush=True,
                )

        train_loss = epoch_loss / epoch_batches
        print(f"epoch={epoch} starting validation batches={len(validation_loader)}")
        try:
            metrics = evaluate(
                model,
                validation_loader,
                device,
                logging_steps=config.evaluation_logging_steps,
                phase="validation",
            )
        except Exception as exc:
            raise RuntimeError(f"Validation failed after epoch {epoch}") from exc
        validation_loss = float(metrics["loss"])
        is_best = validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = validation_loss
            best_model_state = copy_model_state_dict(model)
            best_metrics = copy_metrics(metrics)
            best_epoch = epoch
            best_global_step = global_step
            best_train_loss = train_loss

        epoch_log = prefix_metrics(metrics, "validation") | {
            "train/epoch_loss": train_loss,
            "train/epoch": float(epoch),
            "train/global_step": global_step,
            "epoch": epoch,
        }
        if is_best:
            epoch_log |= prefix_metrics(metrics, "best") | {
                "best/epoch": epoch,
                "best/global_step": global_step,
                "best/train_loss": train_loss,
            }
        wandb.log(epoch_log, step=global_step)

        best_marker = " best" if is_best else ""
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"validation_loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f} {format_per_class_f1(metrics)} "
            f"lr={lr_scheduler.get_last_lr()[0]:.2e}{best_marker}"
        )

    if best_model_state is None or best_metrics is None:
        raise RuntimeError("Training completed without selecting a best model.")

    model.load_state_dict(best_model_state)
    save_final_model(model, feature_extractor, config.output_dir)
    wandb.log(
        prefix_metrics(best_metrics, "chosen")
        | {
            "chosen/epoch": best_epoch,
            "chosen/global_step": best_global_step,
            "chosen/train_loss": best_train_loss,
            "epoch": best_epoch,
        },
        step=global_step,
    )
    print(
        f"saved final model to {config.output_dir} "
        f"chosen_epoch={best_epoch} chosen_step={best_global_step} "
        f"train_loss={best_train_loss:.4f} validation_loss={best_metrics['loss']:.4f} "
        f"accuracy={best_metrics['accuracy']:.4f} f1_macro={best_metrics['f1_macro']:.4f} "
        f"{format_per_class_f1(best_metrics)}"
    )


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def memory_summary(device: torch.device) -> str:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_gib = max_rss / 2**30 if sys.platform == "darwin" else max_rss / 2**20  # macOS reports bytes, Linux KiB
    parts = [f"max_rss={max_rss_gib:.2f}GiB"]
    if device.type == "mps":
        parts.append(f"mps_allocated={torch.mps.current_allocated_memory() / 2**30:.2f}GiB")
        parts.append(f"mps_driver={torch.mps.driver_allocated_memory() / 2**30:.2f}GiB")
    elif device.type == "cuda":
        parts.append(f"cuda_allocated={torch.cuda.memory_allocated() / 2**30:.2f}GiB")
        parts.append(f"cuda_reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB")
    return " ".join(parts)


def release_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


def copy_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def copy_metrics(metrics: dict[str, float | list[float]]) -> dict[str, float | list[float]]:
    return {name: value.copy() if isinstance(value, list) else float(value) for name, value in metrics.items()}


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
    logging_steps: int = 100,
    phase: str = "evaluation",
) -> dict[str, float | list[float]]:
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    total_frames = 0

    for batch_index, batch in enumerate(data_loader, start=1):
        if logging_steps > 0 and (batch_index == 1 or batch_index % logging_steps == 0):
            print(f"{phase} batch={batch_index}/{len(data_loader)} {memory_summary(device)}", flush=True)

        batch = move_to_device(batch, device)
        with autocast_context(device):
            loss, logits = compute_loss(model, batch)

        labels = batch["labels"]
        mask = labels.sum(-1) > 0
        if not mask.any():
            release_device_cache(device)
            continue

        frame_count = mask.sum().item()
        total_loss += loss.item() * frame_count
        total_frames += frame_count
        all_predictions.append(logits.argmax(-1)[mask].cpu())
        all_labels.append(labels.argmax(-1)[mask].cpu())
        release_device_cache(device)

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


def format_per_class_f1(metrics: dict[str, float | list[float]]) -> str:
    per_class = metrics["f1_per_class"]
    if isinstance(per_class, list) and len(per_class) == 2:
        return f"f1_e={per_class[0]:.4f} f1_j={per_class[1]:.4f}"
    return "f1_e=0.0000 f1_j=0.0000"


def save_final_model(model: torch.nn.Module, feature_extractor: Wav2Vec2FeatureExtractor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
