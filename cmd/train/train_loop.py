import os

import torch
import wandb
from data_collator import FrameLabelCollator
from dataset import test_data, training_data, validation_data
from dotenv import load_dotenv
from model_setup import feature_extractor, model
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional
from transformers import Trainer, TrainingArguments

load_dotenv()
os.environ.setdefault("WANDB_PROJECT", "janglish")

if len(training_data) == 0:
    raise RuntimeError("No training examples found. Check data/data-pairs.csv and matching .flac/.json files.")


class LidTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        mask = labels.sum(-1) > 0
        targets = labels.argmax(-1)
        if mask.any():
            loss = functional.cross_entropy(logits[mask], targets[mask])
        else:
            loss = logits.sum() * 0

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: (B, T, C), labels: (B, T, C)
    preds = logits.argmax(-1)
    mask = labels.sum(-1) > 0
    labels = labels.argmax(-1)
    preds_flat = preds[mask]
    labels_flat = labels[mask]
    if len(labels_flat) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0}

    return {
        "accuracy": accuracy_score(labels_flat, preds_flat),
        "f1_macro": f1_score(labels_flat, preds_flat, average="macro", labels=[0, 1], zero_division=0),
        "f1_per_class": f1_score(labels_flat, preds_flat, average=None, labels=[0, 1], zero_division=0).tolist(),
    }


args = TrainingArguments(
    output_dir="./models/lid-wav2vec2",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # effective batch 8; tune to your VRAM
    learning_rate=3e-5,  # standard for wav2vec2 fine-tuning
    warmup_steps=4,
    num_train_epochs=20,
    eval_strategy="epoch",
    save_strategy="best",
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=5,
    report_to="wandb",
    run_name="lid-wav2vec2",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    remove_unused_columns=False,  # critical — Trainer will drop "audio" otherwise
)

trainer = LidTrainer(
    model=model,
    args=args,
    train_dataset=training_data,
    eval_dataset=validation_data,
    data_collator=FrameLabelCollator(feature_extractor),
    compute_metrics=compute_metrics,
)
try:
    trainer.train()
    trainer.evaluate(eval_dataset=test_data, metric_key_prefix="test")
finally:
    if wandb.run is not None:
        wandb.finish()
