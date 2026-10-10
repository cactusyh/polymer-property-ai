#!/usr/bin/env python
"""Short end-to-end GPU fine-tuning smoke test for chain band gap."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.psmiles_tokenizer import tokenize_psmiles  # noqa: E402
from src.models.psmiles_bert import PSmilesBert, PSmilesBertConfig, RegressionHead  # noqa: E402

MAX_LENGTH = 192
PROPERTY = "band_gap_chain"


class PolymerDataset(Dataset):
    def __init__(self, strings: list[str], labels: np.ndarray, vocab: dict[str, int], max_length: int):
        encoded_ids, encoded_masks = [], []
        for text in strings:
            ids = [vocab["<CLS>"]] + [vocab.get(token, vocab["<UNK>"]) for token in tokenize_psmiles(text)] + [vocab["<SEP>"]]
            ids = ids[:max_length]
            mask = [1] * len(ids)
            ids += [vocab["<PAD>"]] * (max_length - len(ids))
            mask += [0] * (max_length - len(mask))
            encoded_ids.append(ids)
            encoded_masks.append(mask)
        self.ids = torch.tensor(encoded_ids, dtype=torch.long)
        self.masks = torch.tensor(encoded_masks, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.ids[index], self.masks[index], self.labels[index]


def checkpoint_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    config = PSmilesBertConfig(
        vocab_size=len(checkpoint["vocab"]),
        max_sequence_length=int(state["pos.weight"].shape[0]),
        layers=sum(key.startswith("encoder.layers.") and key.endswith("self_attn.in_proj_weight") for key in state),
        hidden_size=int(state["token.weight"].shape[1]),
        attention_heads=8,
        intermediate_size=int(state["encoder.layers.0.linear1.weight"].shape[0]),
        dropout=0.1,
    )
    encoder = PSmilesBert(config)
    encoder.load_state_dict(state)
    return encoder, checkpoint["vocab"], config


def gpu_snapshot() -> dict[str, object]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        utilization, used, total = [value.strip() for value in output.split(",")]
        return {"gpu_utilization_percent": float(utilization), "gpu_memory_used_mib": float(used), "gpu_memory_total_mib": float(total)}
    except Exception as exc:
        return {"gpu_snapshot_error": str(exc)}


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, dtype: torch.dtype) -> float:
    model.eval()
    losses = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        for ids, masks, labels in loader:
            _, pooled = model.encoder(ids.to(device, non_blocking=True), masks.to(device, non_blocking=True))
            prediction = model.head(pooled).squeeze(-1)
            losses.append(float(nn.functional.mse_loss(prediction, labels.to(device, non_blocking=True)).cpu()))
    return float(np.mean(losses))


def run_model(name: str, checkpoint_path: Path, train_strings: list[str], train_labels: np.ndarray, val_strings: list[str], val_labels: np.ndarray, vocab: dict[str, int], config: PSmilesBertConfig, device: torch.device, steps: int, output_dir: Path, seed: int) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if name == "A":
        encoder = PSmilesBert(config)
    else:
        encoder, _, _ = checkpoint_model(checkpoint_path)
    model = nn.ModuleDict({"encoder": encoder, "head": RegressionHead(config.hidden_size, 1)}).to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = True
    trainable_encoder = sum(parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad)
    train_dataset = PolymerDataset(train_strings, train_labels, vocab, config.max_sequence_length)
    val_dataset = PolymerDataset(val_strings, val_labels, vocab, config.max_sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=0)
    target_mean = float(train_labels.mean())
    target_std = float(train_labels.std() or 1.0)
    train_dataset.labels = (train_dataset.labels - target_mean) / target_std
    val_dataset.labels = (val_dataset.labels - target_mean) / target_std
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    dtype = torch.bfloat16
    initial_validation_loss = evaluate_loss(model, val_loader, device, dtype)
    iterator = iter(train_loader)
    step_records = []
    checkpoint_step = max(1, min(100, steps - 20))
    reload_path = output_dir / name / "step_reload.pt"
    output_dir.joinpath(name).mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        try:
            ids, masks, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            ids, masks, labels = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, pooled = model.encoder(ids.to(device, non_blocking=True), masks.to(device, non_blocking=True))
            prediction = model.head(pooled).squeeze(-1)
            loss = nn.functional.mse_loss(prediction, labels.to(device, non_blocking=True))
        loss.backward()
        gradient_count = sum(parameter.numel() for parameter in model.encoder.parameters() if parameter.grad is not None)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        step_records.append({"step": step, "loss": float(loss.detach().cpu()), "step_seconds": elapsed, "samples_per_second": len(labels) / elapsed, "encoder_parameters_with_grad": gradient_count, **gpu_snapshot()})
        if step == checkpoint_step:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "vocab": vocab, "config": config.__dict__, "target_mean": target_mean, "target_std": target_std}, reload_path)
            reloaded = nn.ModuleDict({"encoder": PSmilesBert(config), "head": RegressionHead(config.hidden_size, 1)}).to(device)
            saved = torch.load(reload_path, map_location=device, weights_only=False)
            reloaded.load_state_dict(saved["model"])
            resumed_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=2e-5, weight_decay=0.01)
            resumed_optimizer.load_state_dict(saved["optimizer"])
            model, optimizer = reloaded, resumed_optimizer
    final_validation_loss = evaluate_loss(model, val_loader, device, dtype)
    final_path = output_dir / name / "final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": steps, "vocab": vocab, "config": config.__dict__, "target_mean": target_mean, "target_std": target_std}, final_path)
    return {"model": name, "checkpoint_source": str(checkpoint_path) if name != "A" else "random_initialization", "steps": steps, "train_count": len(train_strings), "validation_count": len(val_strings), "initial_validation_loss_normalized": initial_validation_loss, "final_validation_loss_normalized": final_validation_loss, "trainable_encoder_parameters": trainable_encoder, "encoder_parameters_with_grad_last_step": step_records[-1]["encoder_parameters_with_grad"], "encoder_gradient_reached": step_records[-1]["encoder_parameters_with_grad"] == trainable_encoder, "reload_checkpoint": str(reload_path), "final_checkpoint": str(final_path), "mean_step_seconds": float(np.mean([row["step_seconds"] for row in step_records])), "mean_samples_per_second": float(np.mean([row["samples_per_second"] for row in step_records])), "max_gpu_memory_used_mib": float(max(row.get("gpu_memory_used_mib", 0.0) for row in step_records)), "mean_gpu_utilization_percent": float(np.mean([row.get("gpu_utilization_percent", 0.0) for row in step_records])), "step_records": step_records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to run the GPU smoke test")
    device = torch.device("cuda:0")
    registry = pd.read_parquet(ROOT / "data/processed/electronic_property_registry.parquet")
    splits = pd.read_csv(ROOT / "data/splits/electronic_property_splits.csv")
    data = registry[(registry.property_name == PROPERTY) & registry.valid_psmiles & registry.canonical_psmiles.notna() & registry.target_value.notna()].drop_duplicates("canonical_psmiles").copy()
    split = splits[(splits.property_name == PROPERTY) & (splits.seed == args.seed)].drop_duplicates("canonical_psmiles")
    train_ids = set(split.loc[split.random_split == "train", "canonical_psmiles"])
    validation_ids = set(split.loc[split.random_split == "validation", "canonical_psmiles"])
    train = data[data.canonical_psmiles.isin(train_ids)].sort_values("canonical_psmiles").head(512)
    validation = data[data.canonical_psmiles.isin(validation_ids)].sort_values("canonical_psmiles")
    source_encoder, vocab, config = checkpoint_model(ROOT / "checkpoints/B_MLM/best.pt")
    del source_encoder
    output_dir = ROOT / "checkpoints/GPU_SMOKE_CHAIN"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, path in [("B", ROOT / "checkpoints/B_MLM/best.pt"), ("C", ROOT / "checkpoints/C_PERIODIC/best.pt")]:
        results.append(run_model(name, path, train.canonical_psmiles.tolist(), train.target_value.to_numpy(float), validation.canonical_psmiles.tolist(), validation.target_value.to_numpy(float), vocab, config, device, args.steps, output_dir, args.seed))
    summary = [{key: value for key, value in result.items() if key != "step_records"} for result in results]
    pd.DataFrame(summary).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "telemetry.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "train_count": len(train), "validation_count": len(validation), "results": summary}, indent=2))


if __name__ == "__main__":
    main()
