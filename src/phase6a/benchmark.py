"""Small deterministic benchmark utilities; scientific runs use real manifests."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from src.phase6a.contract import TASKS, canonical_json
from src.phase6a.models import MorganEncoder, UnifiedPolymerModel


def regression_metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    from scipy.stats import spearmanr
    y, predicted = np.asarray(y, float), np.asarray(predicted, float)
    valid = np.isfinite(y) & np.isfinite(predicted)
    y, predicted = y[valid], predicted[valid]
    if not len(y):
        raise ValueError("No evaluable samples")
    residual = y - predicted
    total = np.square(y - y.mean()).sum()
    return {
        "RMSE": float(np.sqrt(np.square(residual).mean())),
        "MAE": float(np.abs(residual).mean()),
        "R2": float(1 - np.square(residual).sum() / total) if total > 0 else float("nan"),
        "Spearman": float(spearmanr(y, predicted).statistic) if len(y) > 2 else float("nan"),
        "n_evaluated": len(y),
    }


def checkpoint_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def matrix(seed: int = 13) -> list[dict[str, object]]:
    dimensions = itertools.product(
        ("morgan", "random_psmiles", "pi1m_mlm"),
        ("structure_only", "chain_md_observed", "chain_md_predicted"),
        (False, True), ("gated", "concat"), ("random", "ood"),
        ("clean-transfer", "production"),
    )
    return [{"encoder": a, "inference_mode": b, "auxiliary_md_loss": c, "fusion": d,
             "split": e, "pretraining_policy": f, "seed": seed}
            for a, b, c, d, e, f in dimensions]


def smoke_train(output: Path, manifest_hash: str, seed: int = 13) -> dict[str, object]:
    """Synthetic software fixture only; never a scientific performance result."""
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    texts = ["*CC*", "*CO*", "*CCC*", "*C=C*", "*CN*", "*CCO*", "*C(Cl)C*", "*C(F)C*"]
    model = UnifiedPolymerModel(MorganEncoder(hidden_size=16), hidden_size=12,
                                modality_dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    values = torch.randn(len(texts), 8)
    availability = torch.rand(len(texts), 8) > 0.2
    targets = {task: torch.linspace(0, 1, len(texts)) for task in TASKS}
    masks = {task: torch.tensor([i % (j + 2) != 0 for i in range(len(texts))])
             for j, task in enumerate(TASKS)}
    model.train()
    losses = []
    for _ in range(2):
        result = model(texts, "chain_md_observed", values, availability)
        loss, counts = model.properties.masked_loss(result["predictions"], targets, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        structure = model(texts, "structure_only")
        predicted = model(texts, "chain_md_predicted")
    record = {
        "result_type": "CPU_SOFTWARE_SMOKE_TEST_SYNTHETIC_NON_SCIENTIFIC",
        "seed": seed, "n_fixture_rows": len(texts), "train_steps": len(losses),
        "losses": losses, "valid_train_samples_by_task": counts,
        "structure_only_shape": list(structure["embedding"].shape),
        "predicted_md_shape": list(predicted["embedding"].shape),
        "checkpoint_hash": checkpoint_hash(model), "data_manifest_hash": manifest_hash,
        "scientific_performance": None,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke.json").write_bytes(canonical_json(record) + b"\n")
    return record


def result_record(*, task: str, metrics: dict[str, float | int], seed: int,
                  split: str, checkpoint_sha256: str, manifest_sha256: str,
                  result_type: str, configuration: dict[str, object]) -> dict[str, object]:
    if task not in TASKS or result_type not in (
        "CPU_SOFTWARE_SMOKE_TEST_SYNTHETIC_NON_SCIENTIFIC",
        "CLEAN_TRANSFER_BENCHMARK", "PRODUCTION_RESULT",
    ):
        raise ValueError("Unrecognized task or result type")
    return {"task": task, **metrics, "seed": seed, "split": split,
            "checkpoint_sha256": checkpoint_sha256, "data_manifest_sha256": manifest_sha256,
            "result_type": result_type, "configuration": json.loads(canonical_json(configuration))}
