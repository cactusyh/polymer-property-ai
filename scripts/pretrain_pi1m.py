#!/usr/bin/env python
"""GPU-aware, resumable PI1M masked-language pretraining."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.psmiles_tokenizer import tokenize_psmiles
from src.models.psmiles_bert import MLMHead, PSmilesBert, PSmilesBertConfig

SPECIAL = ["<PAD>", "<MASK>", "<UNK>", "<CLS>", "<SEP>"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vocab_from_pi(path):
    c = collections.Counter()
    for s in pd.read_parquet(path, columns=["raw_psmiles"]).raw_psmiles.astype(str):
        c.update(tokenize_psmiles(s))
    return {
        t: i for i, t in enumerate(SPECIAL + sorted(t for t in c if t not in SPECIAL))
    }


def encode(s, vocab, maxlen):
    ids = (
        [vocab["<CLS>"]]
        + [vocab.get(t, vocab["<UNK>"]) for t in tokenize_psmiles(str(s))]
        + [vocab["<SEP>"]]
    )
    ids = ids[:maxlen]
    mask = [1] * len(ids)
    ids += [0] * (maxlen - len(ids))
    mask += [0] * (maxlen - len(mask))
    return torch.tensor(ids), torch.tensor(mask)


class SmilesDS(Dataset):
    def __init__(self, path, vocab, maxlen, limit=None):
        self.df = (
            pd.read_parquet(path, columns=["raw_psmiles", "split"])
            .query('split == "train"')
            .head(limit)
        )
        self.vocab = vocab
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return encode(self.df.iloc[i].raw_psmiles, self.vocab, self.maxlen)


class TokenCacheDS(Dataset):
    """Dataset backed by persistent token IDs; no PSMILES parsing in workers."""

    def __init__(self, cache, train_path, maxlen, limit=None):
        cache = Path(cache)
        self.ids = np.load(cache / "token_ids.npy", mmap_mode="r")
        self.lengths = np.load(cache / "lengths.npy", mmap_mode="r")
        self.indices = np.load(cache / "train_indices.npy", mmap_mode="r")
        if self.ids.shape[0] != 988561 or self.lengths.shape[0] != self.ids.shape[0]:
            raise RuntimeError("token cache row count/lengths mismatch")
        if len(self.indices) != 968879:
            raise RuntimeError("token cache train split row count mismatch")
        if int(self.indices.max()) >= self.ids.shape[0]:
            raise RuntimeError("token cache train indices out of range")
        parquet_rows = len(pd.read_parquet(train_path, columns=["split"]))
        if parquet_rows != self.ids.shape[0]:
            raise RuntimeError("token cache and corpus row counts differ")
        polymers = pd.read_parquet(train_path, columns=["polymer_id"])[
            "polymer_id"
        ].astype(str)
        identity = hashlib.sha256("\n".join(polymers).encode()).hexdigest()
        recorded = (cache / "row_identity_sha256.txt").read_text().strip()
        if (
            identity != recorded
            or identity
            != "0daa640d36592143f26b0a70f999e2e9bbbf38a737e799bf4748ffb9052f01e4"
        ):
            raise RuntimeError("token cache row identity hash mismatch")
        self.maxlen = maxlen
        self.limit = min(len(self.indices), limit) if limit else len(self.indices)

    def __len__(self):
        return self.limit

    def __getitem__(self, i):
        row = int(self.indices[i])
        return (
            torch.as_tensor(np.asarray(self.ids[row], dtype=np.int64)),
            torch.as_tensor(np.asarray(self.ids[row] != 0, dtype=np.bool_)),
            row,
        )


class EpochPermutationSampler(Sampler):
    """Reproducible sample order derived only from seed and epoch."""

    def __init__(self, n, seed, epoch=0):
        self.n, self.seed, self.epoch = n, int(seed), int(epoch)
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.permutation = torch.randperm(n, generator=g).tolist()
        self.sha256 = hashlib.sha256(
            np.asarray(self.permutation, dtype=np.int64).tobytes()
        ).hexdigest()

    def __iter__(self):
        return iter(self.permutation)

    def __len__(self):
        return self.n


def mask_batch(ids, vocab, p=0.15):
    labels = torch.full_like(ids, -100)
    special = torch.tensor([vocab[x] for x in SPECIAL], device=ids.device)
    can = (ids != 0) & ~torch.isin(ids, special)
    chosen = (torch.rand_like(ids.float()) < p) & can
    labels[chosen] = ids[chosen]
    out = ids.clone()
    out[chosen] = vocab["<MASK>"]
    return out, labels


def atomic_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as f:
        tmp = Path(f.name)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def rng_state(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_state = (
        state["torch"]
        if isinstance(state["torch"], torch.Tensor)
        else torch.tensor(state["torch"], dtype=torch.uint8)
    )
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(
            [
                x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.uint8)
                for x in state["cuda"]
            ]
        )
    loader_state = (
        state["loader"]
        if isinstance(state["loader"], torch.Tensor)
        else torch.tensor(state["loader"], dtype=torch.uint8)
    )
    generator.set_state(loader_state)


def file_sha(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def cache_sha(path):
    h = hashlib.sha256()
    for name in (
        "token_ids.npy",
        "lengths.npy",
        "vocab.json",
        "train_indices.npy",
        "row_identity_sha256.txt",
    ):
        h.update((Path(path) / name).read_bytes())
    return h.hexdigest()


def object_hash(value):
    h = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            h.update(str(key).encode())
            h.update(object_hash(value[key]).encode())
    elif isinstance(value, (list, tuple)):
        for item in value:
            h.update(object_hash(item).encode())
    else:
        h.update(repr(value).encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_pi1m.yaml")
    ap.add_argument("--train-path")
    ap.add_argument("--output-dir")
    ap.add_argument("--token-cache")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-interval", type=int, default=100)
    args = ap.parse_args()
    audit_dir = (
        Path(os.environ["PHASE6X_AUDIT_DIR"])
        if os.environ.get("PHASE6X_AUDIT_DIR")
        else None
    )
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke and not args.token_cache:
        raise RuntimeError("--token-cache is mandatory for full training")
    with open(args.config, encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    train_path = Path(args.train_path or cfg["train_path"])
    out = Path(args.output_dir or cfg["output_dir"])
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU training requested but CUDA is unavailable")
    if (
        out.exists()
        and any((out / n).exists() for n in ("last.pt", "best.pt"))
        and not args.resume
    ):
        raise FileExistsError(
            f"output contains checkpoint; use --resume explicitly: {out}"
        )
    seed_all(int(cfg["seed"]))
    cache = Path(args.token_cache) if args.token_cache else None
    vocab = (
        json.loads((cache / "vocab.json").read_text())
        if cache
        else vocab_from_pi(train_path)
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(vocab, sort_keys=True), encoding="utf-8")
    mc = cfg["model"]
    opt_cfg = cfg["training"]
    corpus_sha = file_sha(train_path)
    config_sha = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    cache_hash = cache_sha(cache) if cache else None
    cache_manifest = (
        json.loads((cache / "cache_manifest.json").read_text()) if cache else None
    )
    if cache and cache_hash != cache_manifest["combined_cache_sha256"]:
        raise RuntimeError("token cache combined hash mismatch")
    if cache and cache_manifest["corpus_sha256"] != corpus_sha:
        raise RuntimeError("token cache source corpus hash mismatch")
    vocab_sha = (
        file_sha(cache / "vocab.json") if cache else file_sha(out / "vocab.json")
    )
    if (
        cache
        and vocab_sha
        != "c8e878d8339041a098e6507ec5fa2a0ce8e9633c2f2771582cfb929eff4e5869"
    ):
        raise RuntimeError("token cache vocabulary hash mismatch")
    model = PSmilesBert(PSmilesBertConfig(vocab_size=len(vocab), **mc)).to(device)
    head = MLMHead(mc["hidden_size"], len(vocab)).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 0.01),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    step = 0
    start_epoch = 0
    resume_path = out / "last.pt"
    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint missing: {resume_path}")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        expected = {
            "corpus_sha256": corpus_sha,
            "config_sha256": config_sha,
            "training_commit": commit,
            "cache_sha256": cache_hash,
            "vocabulary_sha256": vocab_sha,
        }
        if any(state.get(k) != v for k, v in expected.items()):
            raise RuntimeError(
                "resume metadata is incompatible with current corpus/config/commit"
            )
        model.load_state_dict(state["model"])
        head.load_state_dict(state["head"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {}))
        step, start_epoch = state["step"], state["epoch"]
    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]))
    loader_generator = torch.Generator().manual_seed(int(cfg["seed"]) + 100003)
    resume_batch = 0
    ds = (
        TokenCacheDS(cache, train_path, mc["max_sequence_length"], args.limit)
        if cache
        else SmilesDS(train_path, vocab, mc["max_sequence_length"], args.limit)
    )
    sampler = EpochPermutationSampler(len(ds), int(cfg["seed"])) if cache else None
    rows = []
    model.train()
    head.train()
    limit = args.max_steps or 0
    if args.resume:
        restore_rng(state["rng_state"], generator)
        resume_batch = state.get("batch_index", -1) + 1
    for ep in range(start_epoch, cfg["training"]["epochs"]):
        if sampler:
            sampler.epoch = ep
            sampler.__init__(len(ds), int(cfg["seed"]), ep)
        loader_generator = torch.Generator().manual_seed(int(cfg["seed"]) + 100003 + ep)
        dl = DataLoader(
            ds,
            batch_size=cfg["training"].get("batch_size", 16),
            shuffle=(not args.smoke and sampler is None),
            sampler=sampler,
            generator=loader_generator,
            pin_memory=device.type == "cuda",
            num_workers=0
            if (args.smoke or sampler is not None)
            else (2 if device.type == "cuda" else 0),
        )
        for batch_index, batch in enumerate(dl):
            ids, mask = batch[:2]
            batch_rows = batch[2] if len(batch) > 2 else None
            if ep == start_epoch and batch_index < resume_batch:
                continue
            if limit and step >= limit:
                break
            step_start = time.perf_counter()
            ids, mask = (
                ids.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
            )
            ids_m, labels = mask_batch(ids, vocab, cfg["training"]["mask_probability"])
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                x, _ = model(ids_m, mask)
                logits = head(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, len(vocab)), labels.view(-1), ignore_index=-100
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()),
                cfg["training"].get("max_grad_norm", 1.0),
            )
            scaler.step(opt)
            scaler.update()
            step += 1
            if audit_dir:
                state_rng = rng_state(loader_generator)
                audit = {
                    "step": step,
                    "epoch": ep,
                    "batch_index": batch_index,
                    "batch_row_identity_hash": object_hash(batch_rows),
                    "permutation_sha256": sampler.sha256 if sampler else None,
                    "masked_positions_hash": object_hash(labels != -100),
                    "masked_input_hash": object_hash(ids_m),
                    "label_hash": object_hash(labels),
                    "loss": float(loss.detach().cpu()),
                    "accuracy": float(
                        (logits.argmax(-1)[labels != -100] == labels[labels != -100])
                        .float()
                        .mean()
                    )
                    if (labels != -100).any()
                    else 0.0,
                    "model_hash": object_hash(model.state_dict()),
                    "head_hash": object_hash(head.state_dict()),
                    "optimizer_hash": object_hash(opt.state_dict()),
                    "scaler_hash": object_hash(scaler.state_dict()),
                    "rng_hash": object_hash(state_rng),
                    "python_rng_hash": object_hash(state_rng["python"]),
                    "numpy_rng_hash": object_hash(state_rng["numpy"]),
                    "torch_rng_hash": object_hash(state_rng["torch"]),
                    "cuda_rng_hash": object_hash(state_rng["cuda"]),
                    "dataloader_generator_rng_hash": object_hash(state_rng["loader"]),
                    "logits_hash": object_hash(logits),
                    "prediction_hash": object_hash(logits.argmax(-1)),
                }
                (audit_dir / f"step_{step:04d}.json").write_text(
                    json.dumps(audit, sort_keys=True) + "\n"
                )
            masked = labels != -100
            acc = (
                float((logits.argmax(-1)[masked] == labels[masked]).float().mean())
                if masked.any()
                else 0.0
            )
            rows.append(
                {
                    "step": step,
                    "epoch": ep,
                    "batch_index": batch_index,
                    "train_mlm_loss": float(loss.detach().cpu()),
                    "masked_token_accuracy": acc,
                    "step_seconds": time.perf_counter() - step_start,
                    "tokenization_calls": 0 if cache else len(ids),
                    "batch_row_identity_hash": hashlib.sha256(
                        np.asarray(
                            batch_rows if batch_rows is not None else [], dtype=np.int64
                        ).tobytes()
                    ).hexdigest(),
                    "input_cuda": bool(ids.is_cuda),
                    "loss_cuda": bool(loss.is_cuda),
                    "grad_cuda": bool(next(model.parameters()).grad.is_cuda),
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(0)
                    if device.type == "cuda"
                    else None,
                    "peak_memory_bytes": int(
                        torch.cuda.max_memory_allocated()
                        if device.type == "cuda"
                        else 0
                    ),
                }
            )
            checkpoint = {
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "config": mc,
                "vocab": vocab,
                "step": step,
                "epoch": ep,
                "batch_index": batch_index,
                "device": str(device),
                "corpus_sha256": corpus_sha,
                "config_sha256": config_sha,
                "training_commit": commit,
                "cache_sha256": cache_hash,
                "vocabulary_sha256": vocab_sha,
                "rng_state": rng_state(loader_generator),
                "cache_manifest": cache_manifest,
                "checkpoint_interval": args.checkpoint_interval,
                "sampler_seed": int(cfg["seed"]),
                "sampler_epoch": ep,
                "permutation_sha256": sampler.sha256 if sampler else None,
            }
            if args.smoke or step % max(1, args.checkpoint_interval) == 0:
                atomic_save(checkpoint, resume_path)
        if limit and step >= limit:
            break
    final_epoch = ep if rows else start_epoch
    final_batch = batch_index if rows else resume_batch - 1
    pd.DataFrame(rows).to_csv(out / "training_curves.csv", index=False)
    atomic_save(
        {
            "model": model.state_dict(),
            "head": head.state_dict(),
            "config": mc,
            "vocab": vocab,
            "step": step,
            "epoch": final_epoch,
            "batch_index": final_batch,
            "device": str(device),
            "corpus_sha256": corpus_sha,
            "config_sha256": config_sha,
            "training_commit": commit,
            "cache_sha256": cache_hash,
            "vocabulary_sha256": vocab_sha,
            "rng_state": rng_state(loader_generator),
            "cache_manifest": cache_manifest,
            "checkpoint_interval": args.checkpoint_interval,
            "sampler_seed": int(cfg["seed"]),
            "sampler_epoch": final_epoch,
            "permutation_sha256": sampler.sha256 if sampler else None,
        },
        out / "best.pt",
    )
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "corpus_sha256": corpus_sha,
                "config_sha256": config_sha,
                "training_commit": commit,
                "cache_sha256": cache_hash,
                "vocabulary_sha256": vocab_sha,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "final_epoch": final_epoch,
                "final_batch_index": final_batch,
                "global_step": step,
                "sampler_seed": int(cfg["seed"]),
                "sampler_epoch": final_epoch,
                "permutation_sha256": sampler.sha256 if sampler else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "steps": step,
                "output": str(out),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "peak_memory_bytes": int(
                    torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
                ),
                "token_cache": str(cache) if cache else None,
            }
        )
    )


if __name__ == "__main__":
    main()
