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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.psmiles_tokenizer import tokenize_psmiles
from src.models.psmiles_bert import MLMHead, PSmilesBert, PSmilesBertConfig

SPECIAL = ["<PAD>", "<MASK>", "<UNK>", "<CLS>", "<SEP>"]
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def vocab_from_pi(path):
    c = collections.Counter()
    for s in pd.read_parquet(path, columns=["raw_psmiles"]).raw_psmiles.astype(str): c.update(tokenize_psmiles(s))
    return {t: i for i, t in enumerate(SPECIAL + sorted(t for t in c if t not in SPECIAL))}
def encode(s, vocab, maxlen):
    ids = [vocab["<CLS>"]] + [vocab.get(t, vocab["<UNK>"]) for t in tokenize_psmiles(str(s))] + [vocab["<SEP>"]]; ids = ids[:maxlen]; mask = [1] * len(ids); ids += [0] * (maxlen - len(ids)); mask += [0] * (maxlen - len(mask)); return torch.tensor(ids), torch.tensor(mask)
class SmilesDS(Dataset):
    def __init__(self, path, vocab, maxlen, limit=None): self.df = pd.read_parquet(path, columns=["raw_psmiles", "split"]).query('split == "train"').head(limit); self.vocab = vocab; self.maxlen = maxlen
    def __len__(self): return len(self.df)
    def __getitem__(self, i): return encode(self.df.iloc[i].raw_psmiles, self.vocab, self.maxlen)
def mask_batch(ids, vocab, p=0.15):
    labels = torch.full_like(ids, -100); special = torch.tensor([vocab[x] for x in SPECIAL], device=ids.device); can = (ids != 0) & ~torch.isin(ids, special); chosen = (torch.rand_like(ids.float()) < p) & can; labels[chosen] = ids[chosen]; out = ids.clone(); out[chosen] = vocab["<MASK>"]; return out, labels
def atomic_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as f: tmp = Path(f.name)
    try: torch.save(value, tmp); os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()
def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "loader": generator.get_state()}
def restore_rng(state, generator):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); cpu_state = state["torch"] if isinstance(state["torch"], torch.Tensor) else torch.tensor(state["torch"], dtype=torch.uint8); torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and state.get("cuda"): torch.cuda.set_rng_state_all([x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.uint8) for x in state["cuda"]])
    loader_state = state["loader"] if isinstance(state["loader"], torch.Tensor) else torch.tensor(state["loader"], dtype=torch.uint8); generator.set_state(loader_state)
def file_sha(path):
    h = hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/pretrain_pi1m.yaml"); ap.add_argument("--train-path"); ap.add_argument("--output-dir"); ap.add_argument("--max-steps", type=int); ap.add_argument("--limit", type=int); ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume", action="store_true"); args = ap.parse_args()
    with open(args.config, encoding="utf-8") as config_file: cfg = yaml.safe_load(config_file)
    train_path = Path(args.train_path or cfg["train_path"]); out = Path(args.output_dir or cfg["output_dir"]); use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()); device = torch.device("cuda" if use_cuda else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("GPU training requested but CUDA is unavailable")
    if out.exists() and any((out / n).exists() for n in ("last.pt", "best.pt")) and not args.resume: raise FileExistsError(f"output contains checkpoint; use --resume explicitly: {out}")
    seed_all(int(cfg["seed"])); vocab = vocab_from_pi(train_path); out.mkdir(parents=True, exist_ok=True); (out / "vocab.json").write_text(json.dumps(vocab, sort_keys=True), encoding="utf-8"); mc = cfg["model"]; opt_cfg = cfg["training"]; corpus_sha = file_sha(train_path); config_sha = hashlib.sha256(Path(args.config).read_bytes()).hexdigest(); commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(); model = PSmilesBert(PSmilesBertConfig(vocab_size=len(vocab), **mc)).to(device); head = MLMHead(mc["hidden_size"], len(vocab)).to(device); opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0.01)); scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda"); step = 0; start_epoch = 0
    resume_path = out / "last.pt"
    if args.resume:
        if not resume_path.exists(): raise FileNotFoundError(f"resume checkpoint missing: {resume_path}")
        state = torch.load(resume_path, map_location="cpu", weights_only=False); expected = {"corpus_sha256": corpus_sha, "config_sha256": config_sha, "training_commit": commit};
        if any(state.get(k) != v for k, v in expected.items()): raise RuntimeError("resume metadata is incompatible with current corpus/config/commit")
        model.load_state_dict(state["model"]); head.load_state_dict(state["head"]); opt.load_state_dict(state["optimizer"]); scaler.load_state_dict(state.get("scaler", {})); step, start_epoch = state["step"], state["epoch"]
    generator = torch.Generator(); generator.manual_seed(int(cfg["seed"])); resume_batch = 0
    ds = SmilesDS(train_path, vocab, mc["max_sequence_length"], args.limit); dl = DataLoader(ds, batch_size=cfg["training"].get("batch_size", 16), shuffle=not args.smoke, generator=generator, pin_memory=device.type == "cuda", num_workers=0 if args.smoke else (2 if device.type == "cuda" else 0)); rows = []; model.train(); head.train(); limit = args.max_steps or 0
    if args.resume: restore_rng(state["rng_state"], generator); resume_batch = state.get("batch_index", -1) + 1
    for ep in range(start_epoch, cfg["training"]["epochs"]):
        for batch_index, (ids, mask) in enumerate(dl):
            if ep == start_epoch and batch_index < resume_batch: continue
            if limit and step >= limit: break
            ids, mask = ids.to(device, non_blocking=True), mask.to(device, non_blocking=True); ids_m, labels = mask_batch(ids, vocab, cfg["training"]["mask_probability"]); opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                x, _ = model(ids_m, mask); logits = head(x); loss = nn.functional.cross_entropy(logits.view(-1, len(vocab)), labels.view(-1), ignore_index=-100)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), cfg["training"].get("max_grad_norm", 1.0)); scaler.step(opt); scaler.update(); step += 1; masked = labels != -100; acc = float((logits.argmax(-1)[masked] == labels[masked]).float().mean()) if masked.any() else 0.0; rows.append({"step": step, "epoch": ep, "train_mlm_loss": float(loss.detach().cpu()), "masked_token_accuracy": acc, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "peak_memory_bytes": int(torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)})
            atomic_save({"model": model.state_dict(), "head": head.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "config": mc, "vocab": vocab, "step": step, "epoch": ep, "batch_index": batch_index, "device": str(device), "corpus_sha256": corpus_sha, "config_sha256": config_sha, "training_commit": commit, "rng_state": rng_state(generator)}, resume_path)
        if limit and step >= limit: break
    final_epoch = ep if rows else start_epoch; final_batch = batch_index if rows else resume_batch - 1; pd.DataFrame(rows).to_csv(out / "training_curves.csv", index=False); atomic_save({"model": model.state_dict(), "head": head.state_dict(), "config": mc, "vocab": vocab, "step": step, "epoch": final_epoch, "batch_index": final_batch, "device": str(device), "corpus_sha256": corpus_sha, "config_sha256": config_sha, "training_commit": commit, "rng_state": rng_state(generator)}, out / "best.pt"); (out / "provenance.json").write_text(json.dumps({"corpus_sha256": corpus_sha, "config_sha256": config_sha, "training_commit": commit, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "final_epoch": final_epoch, "final_batch_index": final_batch, "global_step": step}, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"steps": step, "output": str(out), "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "peak_memory_bytes": int(torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)}))
if __name__ == "__main__": main()
