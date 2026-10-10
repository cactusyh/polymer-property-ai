#!/usr/bin/env python
"""Run the first real GPU experiment: B -> C -> D0 -> Tg OOD comparison."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.psmiles_tokenizer import equivalent_repeat_augmentations, tokenize_psmiles
from src.models.psmiles_bert import MLMHead, PSmilesBert, PSmilesBertConfig, RegressionHead


SPECIAL = ["<PAD>", "<MASK>", "<UNK>", "<CLS>", "<SEP>"]
TARGETS = ["log_Rg", "q1", "q2", "relative_shape_anisotropy"]
QUALITY_WEIGHT = {"good": 1.0, "borderline": 0.5, "invalid": 0.0}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_audit() -> dict[str, object]:
    result: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()),
        "nccl_support": bool(torch.distributed.is_nccl_available()),
        "cuda_built": bool(torch.backends.cuda.is_built()),
    }
    if torch.cuda.is_available():
        result["bf16_support"] = bool(torch.cuda.is_bf16_supported())
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "memory_total_gib": round(props.total_memory / 1024**3, 3),
                    "memory_free_gib": round(free / 1024**3, 3),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        result["devices"] = devices
    else:
        result["bf16_support"] = False
        result["devices"] = []
    return result


def gpu_utilization() -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return float(output.strip().splitlines()[0])
    except Exception:
        return None


def build_vocab(strings: Iterable[str]) -> dict[str, int]:
    tokens = set()
    for text in strings:
        tokens.update(tokenize_psmiles(str(text)))
    return {token: index for index, token in enumerate(SPECIAL + sorted(tokens - set(SPECIAL)))}


def encode(text: str, vocab: dict[str, int], max_length: int) -> tuple[list[int], list[int]]:
    ids = [vocab["<CLS>"]]
    ids += [vocab.get(token, vocab["<UNK>"]) for token in tokenize_psmiles(str(text))]
    ids += [vocab["<SEP>"]]
    ids = ids[:max_length]
    mask = [1] * len(ids)
    pad = max_length - len(ids)
    ids += [vocab["<PAD>"]] * pad
    mask += [0] * pad
    return ids, mask


def encode_frame(strings: Iterable[str], vocab: dict[str, int], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode(text, vocab, max_length) for text in strings]
    return torch.tensor([item[0] for item in encoded], dtype=torch.long), torch.tensor([item[1] for item in encoded], dtype=torch.long)


class TokenDataset(Dataset):
    def __init__(self, strings: list[str], vocab: dict[str, int], max_length: int):
        self.ids, self.mask = encode_frame(strings, vocab, max_length)

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ids[index], self.mask[index]


class RegressionDataset(Dataset):
    def __init__(self, strings: list[str], labels: np.ndarray, vocab: dict[str, int], max_length: int):
        self.ids, self.mask = encode_frame(strings, vocab, max_length)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ids[index], self.mask[index], self.labels[index]


def mask_batch(ids: torch.Tensor, vocab: dict[str, int], probability: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.full_like(ids, -100)
    special = torch.tensor([vocab[token] for token in SPECIAL], device=ids.device)
    eligible = (ids != vocab["<PAD>"]) & ~torch.isin(ids, special)
    selected = (torch.rand_like(ids.float()) < probability) & eligible
    labels[selected] = ids[selected]
    masked = ids.clone()
    masked[selected] = vocab["<MASK>"]
    return masked, labels


def batch_to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)


def make_model(vocab_size: int) -> tuple[PSmilesBert, MLMHead]:
    config = PSmilesBertConfig(
        vocab_size=vocab_size,
        layers=8,
        hidden_size=512,
        attention_heads=8,
        intermediate_size=2048,
        max_sequence_length=192,
        dropout=0.1,
    )
    return PSmilesBert(config), MLMHead(config.hidden_size, vocab_size)


def save_checkpoint(path: Path, model: nn.Module, head: nn.Module, optimizer: torch.optim.Optimizer, step: int, epoch: int, vocab: dict[str, int], best_val: float) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "vocab": vocab,
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, head: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    head.load_state_dict(checkpoint["head"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def evaluate_mlm(model: PSmilesBert, head: MLMHead, loader: DataLoader, vocab: dict[str, int], device: torch.device, autocast_dtype: torch.dtype) -> tuple[float, float]:
    model.eval()
    head.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for batch in loader:
            ids, attention = batch_to_device(batch, device)
            masked, labels = mask_batch(ids, vocab)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                hidden, _ = model(masked, attention)
                logits = head(hidden)
                loss = nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), labels.reshape(-1), ignore_index=-100)
            selected = labels != -100
            losses.append(float(loss.detach().cpu()))
            accuracies.append(float((logits.argmax(-1)[selected] == labels[selected]).float().mean().cpu()) if selected.any() else math.nan)
    model.train()
    head.train()
    return float(np.mean(losses)), float(np.nanmean(accuracies))


def train_b_or_c(
    name: str,
    base: Path | None,
    train_strings: list[str],
    validation_strings: list[str],
    out_dir: Path,
    vocab: dict[str, int],
    device: torch.device,
    max_steps: int | None = None,
    data_fraction: float = 1.0,
    periodic: bool = False,
    resume_path: Path | None = None,
) -> dict[str, object]:
    set_seed(13)
    model, mlm_head = make_model(len(vocab))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(mlm_head.parameters()), lr=2e-4, weight_decay=0.01)
    model.to(device)
    mlm_head.to(device)
    if base is not None:
        load_checkpoint(base, model, mlm_head)
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, model, mlm_head, optimizer)
        start_step = int(checkpoint.get("step", 0))
    else:
        start_step = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    max_train = min(len(train_strings), max(1, int(data_fraction * len(train_strings))))
    train_dataset = TokenDataset(train_strings[:max_train], vocab, 192)
    validation_dataset = TokenDataset(validation_strings, vocab, 192)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    iterator = iter(train_loader)
    scaler_dtype = torch.bfloat16
    grad_accumulation = 4
    total_steps = max_steps or math.ceil(len(train_loader) / grad_accumulation) * 3
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda current_step: (current_step + 1) / warmup_steps
        if current_step < warmup_steps
        else 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / max(1, total_steps - warmup_steps))),
    )
    losses: list[float] = []
    val_losses: list[float] = []
    utilization: list[float] = []
    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    best_val = float("inf")
    if resume_path is not None:
        best_val = float(checkpoint.get("best_val", best_val))
    periodic_strings = []
    if periodic:
        for text in train_strings:
            augmentations = equivalent_repeat_augmentations(text, 3)
            for augmented in augmentations:
                periodic_strings.append((text, augmented))
        periodic_strings = periodic_strings[: max(10000, max_steps or 10000)]
        if not periodic_strings:
            raise RuntimeError("periodicity training has no valid repeat augmentations")
    periodic_index = 0
    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_losses = []
        for _ in range(grad_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            ids, attention = batch_to_device(batch, device)
            masked, labels = mask_batch(ids, vocab)
            with torch.autocast(device_type="cuda", dtype=scaler_dtype):
                hidden, pooled = model(masked, attention)
                logits = mlm_head(hidden)
                mlm_loss = nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), labels.reshape(-1), ignore_index=-100)
                loss = mlm_loss
                if periodic:
                    pairs = periodic_strings[periodic_index : periodic_index + ids.shape[0]]
                    periodic_index = (periodic_index + ids.shape[0]) % len(periodic_strings)
                    original_ids, original_attention = encode_frame([pair[0] for pair in pairs], vocab, 192)
                    augmented_ids, augmented_attention = encode_frame([pair[1] for pair in pairs], vocab, 192)
                    _, original_pooled = model(original_ids.to(device), original_attention.to(device))
                    _, augmented_pooled = model(augmented_ids.to(device), augmented_attention.to(device))
                    original_pooled = nn.functional.normalize(original_pooled.float(), dim=-1)
                    augmented_pooled = nn.functional.normalize(augmented_pooled.float(), dim=-1)
                    periodic_logits = original_pooled @ augmented_pooled.T / 0.07
                    periodic_loss = nn.functional.cross_entropy(periodic_logits, torch.arange(len(pairs), device=device))
                    loss = mlm_loss + 0.1 * periodic_loss
                else:
                    periodic_loss = torch.zeros((), device=device)
            (loss / grad_accumulation).backward()
            accum_losses.append(float(loss.detach().cpu()))
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(mlm_head.parameters()), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        losses.append(float(np.mean(accum_losses)))
        if step <= 3 or step % 25 == 0:
            utilization_value = gpu_utilization()
            if utilization_value is not None:
                utilization.append(utilization_value)
        if step % max(1, min(250, total_steps)) == 0 or step == total_steps:
            val_loss, val_accuracy = evaluate_mlm(model, mlm_head, validation_loader, vocab, device, scaler_dtype)
            val_losses.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out_dir / "best.pt", model, mlm_head, optimizer, step, 0, vocab, best_val)
            print(json.dumps({"model": name, "step": step, "train_loss": losses[-1], "val_loss": val_loss, "val_accuracy": val_accuracy}), flush=True)
    elapsed = time.perf_counter() - start_time
    save_checkpoint(out_dir / "final.pt", model, mlm_head, optimizer, step, 0, vocab, best_val)
    pd.DataFrame(
        {
            "step": np.arange(start_step + 1, start_step + len(losses) + 1),
            "train_loss": losses,
        }
    ).to_csv(out_dir / "training_curves.csv", index=False)
    return {
        "name": name,
        "steps": step,
        "train_loss_first_mean": float(np.mean(losses[: min(10, len(losses))])),
        "train_loss_last_mean": float(np.mean(losses[-min(10, len(losses)):])),
        "validation_loss_last": val_losses[-1] if val_losses else None,
        "best_validation_loss": best_val,
        "elapsed_seconds": elapsed,
        "throughput_sequences_per_second": float((len(losses) * grad_accumulation * train_loader.batch_size) / elapsed),
        "gpu_utilization_mean_percent": float(np.mean(utilization)) if utilization else None,
        "gpu_memory_peak_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3),
        "periodic_pair_count": len(periodic_strings),
    }


def load_encoder_checkpoint(path: Path, vocab_size: int, device: torch.device) -> PSmilesBert:
    model, head = make_model(vocab_size)
    load_checkpoint(path, model, head)
    model.to(device)
    model.eval()
    return model


def fit_scalers(md: pd.DataFrame) -> dict[str, tuple[float, float]]:
    scalers = {}
    for target in TARGETS:
        quality = md[f"{target}_quality_class"].fillna("invalid")
        values = pd.to_numeric(md[target], errors="coerce")
        usable = values.notna() & quality.isin(["good", "borderline"])
        scalers[target] = (float(values[usable].mean()), float(values[usable].std(ddof=0) or 1.0))
    return scalers


def physics_batch_loss(model: PSmilesBert, heads: nn.ModuleList, batch: pd.DataFrame, vocab: dict[str, int], scalers: dict[str, tuple[float, float]], device: torch.device) -> torch.Tensor:
    ids, attention = encode_frame(batch.canonical_psmiles.tolist(), vocab, 192)
    _, pooled = model(ids.to(device), attention.to(device))
    losses = []
    for index, target in enumerate(TARGETS):
        values = pd.to_numeric(batch[target], errors="coerce").to_numpy(float)
        quality = batch[f"{target}_quality_class"].fillna("invalid").map(QUALITY_WEIGHT).to_numpy(float)
        valid = np.isfinite(values) & (quality > 0)
        if not valid.any():
            continue
        labels = torch.tensor((values[valid] - scalers[target][0]) / scalers[target][1], dtype=torch.float32, device=device)
        predictions = heads[index](pooled[torch.tensor(valid, device=device)]).squeeze(-1).float()
        weights = torch.tensor(quality[valid], dtype=torch.float32, device=device)
        losses.append(((predictions - labels) ** 2 * weights).sum() / weights.sum().clamp_min(1.0))
    if not losses:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(losses).mean()


def train_d0(c_checkpoint: Path, pi_train: list[str], md: pd.DataFrame, out_dir: Path, vocab: dict[str, int], device: torch.device, steps: int) -> dict[str, object]:
    set_seed(13)
    model, mlm_head = make_model(len(vocab))
    load_checkpoint(c_checkpoint, model, mlm_head)
    model.to(device)
    mlm_head.to(device)
    heads = nn.ModuleList([RegressionHead(512, 1) for _ in TARGETS]).to(device)
    scalers = fit_scalers(md)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 2e-5},
            {"params": heads.parameters(), "lr": 1e-4},
            {"params": mlm_head.parameters(), "lr": 2e-5},
        ],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps))
    train_dataset = TokenDataset(pi_train[: min(len(pi_train), 200000)], vocab, 192)
    replay_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    replay_iterator = iter(replay_loader)
    md_rows = md.sample(frac=1.0, random_state=13).reset_index(drop=True)
    md_cursor = 0
    history = []
    utilization = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        heads.train()
        optimizer.zero_grad(set_to_none=True)
        end = min(md_cursor + 32, len(md_rows))
        md_batch = md_rows.iloc[md_cursor:end]
        md_cursor = 0 if end >= len(md_rows) else end
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            physics_loss = physics_batch_loss(model, heads, md_batch, vocab, scalers, device)
            try:
                replay_batch = next(replay_iterator)
            except StopIteration:
                replay_iterator = iter(replay_loader)
                replay_batch = next(replay_iterator)
            replay_ids, replay_attention = batch_to_device(replay_batch, device)
            replay_masked, replay_labels = mask_batch(replay_ids, vocab)
            replay_hidden, _ = model(replay_masked, replay_attention)
            replay_logits = mlm_head(replay_hidden)
            replay_loss = nn.functional.cross_entropy(replay_logits.reshape(-1, len(vocab)), replay_labels.reshape(-1), ignore_index=-100)
            loss = 0.8 * physics_loss + 0.2 * replay_loss
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()) + list(mlm_head.parameters()), 1.0)
        optimizer.step()
        scheduler.step()
        history.append({"step": step, "loss": float(loss.detach().cpu()), "physics_loss": float(physics_loss.detach().cpu()), "mlm_replay_loss": float(replay_loss.detach().cpu())})
        if step <= 3 or step % 25 == 0:
            utilization_value = gpu_utilization()
            if utilization_value is not None:
                utilization.append(utilization_value)
        if step % 250 == 0 or step == steps:
            print(json.dumps({"model": "D0", **history[-1]}), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mlm_head": mlm_head.state_dict(), "physics_heads": heads.state_dict(), "vocab": vocab, "scalers": scalers}, out_dir / "final.pt")
    pd.DataFrame(history).to_csv(out_dir / "training_curves.csv", index=False)
    pd.DataFrame([{"target": target, "mean": mean, "std": std} for target, (mean, std) in scalers.items()]).to_csv(out_dir / "target_scalers.csv", index=False)
    elapsed = time.perf_counter() - start
    return {
        "name": "D0_GLOBAL_PHYSICS",
        "steps": steps,
        "physics_loss_first_mean": float(np.mean([row["physics_loss"] for row in history[:10]])),
        "physics_loss_last_mean": float(np.mean([row["physics_loss"] for row in history[-10:]])),
        "elapsed_seconds": elapsed,
        "throughput_md_batches_per_second": float(steps / elapsed),
        "gpu_utilization_mean_percent": float(np.mean(utilization)) if utilization else None,
        "gpu_memory_peak_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3),
    }


def fine_tune_tg(checkpoint_path: Path, tg: pd.DataFrame, vocab: dict[str, int], device: torch.device, protocol: str, fraction: float, seed: int, out_path: Path) -> dict[str, object]:
    set_seed(seed)
    model, mlm_head = make_model(len(vocab))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    head = RegressionHead(512, 1).to(device)
    train = tg[tg.ood_split.eq("train")].reset_index(drop=True)
    validation = tg[tg.ood_split.eq("validation")].reset_index(drop=True)
    test = tg[tg.ood_split.eq("test")].reset_index(drop=True)
    rng = np.random.default_rng(seed + int(fraction * 1000))
    if fraction < 1.0:
        train = train.iloc[rng.choice(len(train), size=max(1, int(len(train) * fraction)), replace=False)].reset_index(drop=True)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=5e-5, weight_decay=0.01)

    def make_loader(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        dataset = RegressionDataset(frame.canonical_psmiles.tolist(), frame.Tg.to_numpy(float), vocab, 192)
        return DataLoader(dataset, batch_size=32, shuffle=shuffle, num_workers=2, pin_memory=True)

    train_loader = make_loader(train, True)
    validation_loader = make_loader(validation, False)
    test_loader = make_loader(test, False)

    def evaluate(loader: DataLoader, labels: np.ndarray) -> dict[str, float]:
        model.eval()
        head.eval()
        predictions = []
        with torch.no_grad():
            for ids, attention, _ in loader:
                ids, attention = ids.to(device), attention.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, pooled = model(ids, attention)
                    predictions.extend(head(pooled).squeeze(-1).float().cpu().numpy())
        predictions = np.asarray(predictions)
        return {"RMSE": float(np.sqrt(np.mean((labels - predictions) ** 2))), "MAE": float(np.mean(np.abs(labels - predictions))), "R2": float(1.0 - np.sum((labels - predictions) ** 2) / np.sum((labels - labels.mean()) ** 2))}

    best_val = float("inf")
    best_state = None
    patience = 0
    for epoch in range(1, 101):
        model.train()
        head.train()
        for ids, attention, labels in train_loader:
            ids, attention = ids.to(device), attention.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, pooled = model(ids, attention)
                loss = nn.functional.mse_loss(head(pooled).squeeze(-1).float(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        val = evaluate(validation_loader, validation.Tg.to_numpy(float))
        if val["RMSE"] < best_val:
            best_val = val["RMSE"]
            best_state = {"model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "head": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                break
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        head.load_state_dict(best_state["head"])
    metrics = evaluate(test_loader, test.Tg.to_numpy(float))
    result = {"model": checkpoint_path.parent.name, "protocol": protocol, "fraction": fraction, "seed": seed, "n_train": len(train), **metrics, "epochs": epoch}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        if out_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="/tmp/first_physics_gpu_inputs")
    parser.add_argument("--output", default=".")
    parser.add_argument("--smoke-steps", type=int, default=750)
    parser.add_argument("--b-steps", type=int, default=0)
    parser.add_argument("--c-steps", type=int, default=0)
    parser.add_argument("--d0-steps", type=int, default=5000)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args()
    audit = gpu_audit()
    if not audit["cuda_available"]:
        raise SystemExit("CUDA is unavailable; aborting full GPU training as required")
    if not audit["bf16_support"]:
        raise SystemExit("BF16 is unavailable; aborting BF16 experiment as required")
    print(json.dumps({"gpu_audit": audit}, indent=2), flush=True)

    inputs = Path(args.inputs)
    output = Path(args.output)
    input_audit = pd.read_csv(inputs / "input_audit.csv").iloc[0].to_dict()
    train_df = pd.read_csv(inputs / "pi1m_valid_train.csv")
    validation_df = pd.read_csv(inputs / "pi1m_valid_validation.csv")
    if not train_df.valid.all() or not validation_df.valid.all():
        raise RuntimeError("invalid PI1M rows entered the neural dataset")
    train_strings = train_df.raw_psmiles.astype(str).tolist()
    validation_strings = validation_df.raw_psmiles.astype(str).tolist()
    vocab = build_vocab(train_strings)
    with (output / "checkpoints").mkdir(parents=True, exist_ok=True) if False else open(os.devnull, "w"):
        pass
    device = torch.device("cuda")
    report: dict[str, object] = {"gpu_audit": audit, "input_audit": input_audit, "vocab_size": len(vocab), "valid_train_rows": len(train_strings), "valid_validation_rows": len(validation_strings)}

    if not args.skip_smoke:
        smoke_dir = output / "checkpoints" / "SMOKE"
        smoke_first = train_b_or_c("SMOKE_FIRST", None, train_strings, validation_strings, smoke_dir, vocab, device, max_steps=max(250, args.smoke_steps // 2), data_fraction=0.02)
        smoke_resume = train_b_or_c("SMOKE_RESUME", None, train_strings, validation_strings, smoke_dir, vocab, device, max_steps=args.smoke_steps, data_fraction=0.02, resume_path=smoke_dir / "final.pt")
        report["smoke"] = {"first": smoke_first, "resume": smoke_resume, "loss_decreased": smoke_resume["train_loss_last_mean"] < smoke_resume["train_loss_first_mean"]}

    b_steps = args.b_steps or 0
    if b_steps <= 0:
        b_steps = math.ceil(len(train_strings) / 32 / 16) * 3
    b = train_b_or_c("B_MLM", None, train_strings, validation_strings, output / "checkpoints" / "B_MLM", vocab, device, max_steps=b_steps)
    c_steps = args.c_steps or b_steps
    c = train_b_or_c("C_PERIODIC", output / "checkpoints" / "B_MLM" / "best.pt", train_strings, validation_strings, output / "checkpoints" / "C_PERIODIC", vocab, device, max_steps=c_steps, periodic=True)
    md = pd.read_csv(inputs / "md_d0_ood_pool.csv")
    tg = pd.read_csv(inputs / "tg_valid.csv")
    val_ids = set(tg.loc[tg.ood_split.eq("validation"), "canonical_psmiles"].dropna())
    test_ids = set(tg.loc[tg.ood_split.eq("test"), "canonical_psmiles"].dropna())
    assert not set(md.canonical_psmiles.dropna()) & val_ids
    assert not set(md.canonical_psmiles.dropna()) & test_ids
    d0 = train_d0(output / "checkpoints" / "C_PERIODIC" / "best.pt", train_strings, md, output / "checkpoints" / "D0_GLOBAL_PHYSICS", vocab, device, args.d0_steps)
    report["training"] = {"B_MLM": b, "C_PERIODIC": c, "D0_GLOBAL_PHYSICS": d0}

    results_path = output / "results" / "first_physics_gpu_test.csv"
    if not args.skip_finetune:
        results_path.unlink(missing_ok=True)
        rows = []
        for checkpoint in [output / "checkpoints" / "C_PERIODIC" / "best.pt", output / "checkpoints" / "D0_GLOBAL_PHYSICS" / "final.pt"]:
            for fraction in [1.0, 0.1]:
                for seed in [13, 17, 23]:
                    rows.append(fine_tune_tg(checkpoint, tg, vocab, device, "ood", fraction, seed, results_path))
        result_df = pd.DataFrame(rows)
        summary = result_df.groupby(["model", "fraction"], as_index=False).agg(RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"), MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), R2_mean=("R2", "mean"), R2_std=("R2", "std"))
        summary.to_csv(output / "results" / "first_physics_gpu_test_summary.csv", index=False)
        report["comparison"] = summary.to_dict(orient="records")
        gains = {}
        for fraction in [1.0, 0.1]:
            c_rmse = result_df[(result_df.model == "C_PERIODIC") & (result_df.fraction == fraction)].RMSE.mean()
            d_rmse = result_df[(result_df.model == "D0_GLOBAL_PHYSICS") & (result_df.fraction == fraction)].RMSE.mean()
            gain = (c_rmse - d_rmse) / c_rmse
            gains[str(fraction)] = {"C_RMSE": c_rmse, "D0_RMSE": d_rmse, "relative_gain": gain, "planning_band": "STRONG GO" if gain > 0.08 else "GO" if gain >= 0.05 else "MODIFY" if gain >= 0.02 else "INVESTIGATE"}
        report["relative_physics_gain"] = gains
    report["strict_inductive_assertions"] = {"md_intersects_tg_validation": False, "md_intersects_tg_test": False}
    report["status"] = "completed"
    report_path = output / "reports" / "first_physics_gpu_test.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# First Physics GPU Test", "", "## Scope", "", "This report uses the existing Tg splits, leakage rules, tokenizer, and MD preprocessing outputs without modification.", "", "## GPU Audit", "", "```json", json.dumps(audit, indent=2), "```", "", "## PI1M Final Check", "", f"- `pi1m_no_tg.parquet` rows: {int(input_audit['pi1m_no_tg_rows']):,}.", f"- Retained invalid rows in source corpus: {int(input_audit['pi1m_invalid_rows']):,}.", f"- Valid rows used by neural pretraining: {len(train_strings) + len(validation_strings):,}.", f"- Valid train rows used: {len(train_strings):,}.", f"- Valid validation rows used: {len(validation_strings):,}.", "- Invalid rows used: 0.", "", "## Strict-Inductive Assertions", "", "- `MD_pretrain ∩ Tg_validation = empty`: PASS.", "- `MD_pretrain ∩ Tg_test = empty`: PASS.", "", "## Results", ""]
    if "smoke" in report:
        lines += [f"- MLM smoke loss decreased: `{report['smoke']['loss_decreased']}`.", f"- Smoke/resume checkpoint path: `{output / 'checkpoints' / 'SMOKE'}`.", ""]
    if "comparison" in report:
        lines += ["| model | fraction | RMSE mean | RMSE std | MAE mean | MAE std | R2 mean | R2 std |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in report["comparison"]:
            lines.append(f"| {row['model']} | {row['fraction']} | {row['RMSE_mean']:.4f} | {row['RMSE_std']:.4f} | {row['MAE_mean']:.4f} | {row['MAE_std']:.4f} | {row['R2_mean']:.4f} | {row['R2_std']:.4f} |")
        lines += ["", "### Relative Physics Gain", ""]
        for fraction, gain in report["relative_physics_gain"].items():
            lines.append(f"- OOD {float(fraction):.0%}: C RMSE={gain['C_RMSE']:.4f}, D0 RMSE={gain['D0_RMSE']:.4f}, gain={gain['relative_gain']:.2%}, planning band=`{gain['planning_band']}`.")
    else:
        lines += ["Full comparison was not run.", ""]
    lines += ["", "## Interpretation", "", "- No statistical significance claim is made from three seeds.", "- Backbone contacts and torsion dynamics should not be expanded until this C-versus-D0 comparison is available and reviewed."]
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
