"""Phase 6A structure, masked single-chain MD, and property model components."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.psmiles_tokenizer import tokenize_psmiles
from src.models.psmiles_bert import PSmilesBert, PSmilesBertConfig
from src.phase6a.contract import MD_FEATURES, TASKS


class TrainOnlyNormalizer:
    """Per-feature statistics from explicitly supplied training rows and QC masks."""

    def __init__(self, features: tuple[str, ...]):
        self.features = features
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.count: np.ndarray | None = None

    def fit(self, values: np.ndarray, mask: np.ndarray, *, partition: str) -> TrainOnlyNormalizer:
        if partition != "train":
            raise ValueError("Normalizer may only fit on training data")
        values, mask = np.asarray(values, float), np.asarray(mask, bool)
        if values.shape != mask.shape or values.ndim != 2 or values.shape[1] != len(self.features):
            raise ValueError("Feature shape or mask mismatch")
        observed = mask & np.isfinite(values)
        self.count = observed.sum(axis=0)
        self.mean = np.array([values[observed[:, i], i].mean() if self.count[i] else 0.0
                              for i in range(values.shape[1])])
        self.std = np.array([values[observed[:, i], i].std() if self.count[i] > 1 else 1.0
                             for i in range(values.shape[1])])
        self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer is not fitted")
        values, mask = np.asarray(values, float), np.asarray(mask, bool)
        if values.shape != mask.shape or values.shape[-1] != len(self.features):
            raise ValueError("Feature shape or order mismatch")
        observed = mask & np.isfinite(values)
        scaled = np.where(observed, (np.where(observed, values, self.mean) - self.mean) / self.std, 0.0)
        return scaled.astype(np.float32), observed


class MaskedChainMDEncoder(nn.Module):
    schema_version = "6a.1.0"
    feature_order = tuple(name for name, _ in MD_FEATURES)

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        count = len(self.feature_order)
        self.value = nn.Linear(count, hidden_size)
        self.missingness = nn.Linear(count, hidden_size, bias=False)
        self.qc = nn.Linear(count, hidden_size, bias=False)
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, values: torch.Tensor, availability: torch.Tensor,
                qc_mask: torch.Tensor | None = None) -> torch.Tensor:
        if values.shape != availability.shape or values.shape[-1] != len(self.feature_order):
            raise ValueError("MD input must follow the frozen feature order")
        qc = availability if qc_mask is None else qc_mask.bool()
        observed = availability.bool() & qc & torch.isfinite(values)
        clean = torch.where(observed, values, torch.zeros_like(values))
        encoded = self.value(clean) + self.missingness((~observed).float()) + self.qc(qc.float())
        return self.net(encoded)


class PolymerEncoderAdapter(nn.Module):
    """Adapter around the repository's existing B_MLM PSMILES BERT implementation."""

    def __init__(self, model: PSmilesBert, vocab: dict[str, int], checkpoint_hash: str = "RANDOM_INIT"):
        super().__init__()
        self.model = model
        self.vocab = vocab
        self.checkpoint_hash = checkpoint_hash
        self.output_size = model.cfg.hidden_size

    @classmethod
    def random(cls, vocab: dict[str, int], **model_kwargs) -> PolymerEncoderAdapter:
        return cls(PSmilesBert(PSmilesBertConfig(vocab_size=len(vocab), **model_kwargs)), vocab)

    @classmethod
    def pretrained(cls, path: Path, *, policy: str, excluded_ids_hash: str | None = None) -> PolymerEncoderAdapter:
        if policy not in ("clean-transfer", "production"):
            raise ValueError(policy)
        if policy == "clean-transfer":
            provenance = path.with_suffix(".provenance.json")
            if not provenance.is_file():
                raise ValueError("Clean-transfer checkpoint needs a provenance sidecar")
            import json
            record = json.loads(provenance.read_text())
            if record.get("policy") != policy or record.get("excluded_identities_sha256") != excluded_ids_hash:
                raise ValueError("Clean-transfer checkpoint exclusion manifest mismatch")
            if record.get("checkpoint_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Clean-transfer checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = PSmilesBertConfig(vocab_size=len(payload["vocab"]), **payload["config"])
        model = PSmilesBert(config)
        model.load_state_dict(payload["model"])
        return cls(model, payload["vocab"], hashlib.sha256(path.read_bytes()).hexdigest())

    def encode_structure(self, psmiles: str | list[str], **kwargs) -> torch.Tensor:
        texts = [psmiles] if isinstance(psmiles, str) else list(psmiles)
        limit = kwargs.get("max_length", self.model.cfg.max_sequence_length)
        ids, masks = [], []
        for text in texts:
            tokens = [self.vocab["<CLS>"]] + [self.vocab.get(t, self.vocab["<UNK>"])
                                                   for t in tokenize_psmiles(text)] + [self.vocab["<SEP>"]]
            tokens = tokens[:limit]
            ids.append(tokens + [self.vocab["<PAD>"]] * (limit - len(tokens)))
            masks.append([1] * len(tokens) + [0] * (limit - len(tokens)))
        device = next(self.model.parameters()).device
        _, pooled = self.model(torch.tensor(ids, device=device), torch.tensor(masks, device=device))
        return pooled

    def forward(self, psmiles: str | list[str]) -> torch.Tensor:
        return self.encode_structure(psmiles)


class MorganEncoder(nn.Module):
    """Fixed RDKit Morgan fingerprint followed by a trainable projection."""

    def __init__(self, hidden_size: int = 128, bits: int = 2048):
        super().__init__()
        self.bits = bits
        self.output_size = hidden_size
        self.net = nn.Sequential(nn.Linear(bits, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))
        self.checkpoint_hash = "RANDOM_INIT_MORGAN"

    def encode_structure(self, psmiles: str | list[str], **kwargs) -> torch.Tensor:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        texts = [psmiles] if isinstance(psmiles, str) else list(psmiles)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=self.bits)
        arrays = []
        for text in texts:
            molecule = Chem.MolFromSmiles(text)
            if molecule is None:
                raise ValueError("Invalid PSMILES for Morgan encoder")
            array = np.zeros(self.bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), array)
            arrays.append(array)
        return self.net(torch.as_tensor(np.stack(arrays), device=next(self.parameters()).device))

    def forward(self, psmiles: str | list[str]) -> torch.Tensor:
        return self.encode_structure(psmiles)


class GatedLateFusion(nn.Module):
    """Missing-MD-safe gated fusion with a direct-concatenation ablation."""

    def __init__(self, structure_size: int, md_size: int, feature_count: int = 8,
                 modality_dropout: float = 0.0, ablation: str = "gated"):
        super().__init__()
        if ablation not in ("gated", "concat"):
            raise ValueError(ablation)
        self.ablation = ablation
        self.modality_dropout = modality_dropout
        self.projection = nn.Linear(md_size, structure_size)
        self.gate = nn.Sequential(nn.Linear(structure_size + md_size + feature_count, structure_size), nn.Sigmoid())
        self.concat = nn.Linear(structure_size + md_size, structure_size)

    def forward(self, structure: torch.Tensor, md: torch.Tensor | None,
                md_mask: torch.Tensor | None, mode: str = "structure_only") -> torch.Tensor:
        if mode == "structure_only" or md is None or md_mask is None:
            return structure
        if mode not in ("chain_md_observed", "chain_md_predicted"):
            raise ValueError(mode)
        present = md_mask.bool().any(dim=-1, keepdim=True).float()
        if self.training and self.modality_dropout:
            present = present * (torch.rand_like(present) >= self.modality_dropout).float()
        if self.ablation == "concat":
            combined = self.concat(torch.cat([structure, md], dim=-1))
            return structure + present * (combined - structure)
        gate = self.gate(torch.cat([structure, md, md_mask.float()], dim=-1))
        return structure + present * gate * self.projection(md)


class MultiTaskPropertyHeads(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.heads = nn.ModuleDict({task: nn.Linear(hidden_size, 1) for task in TASKS})

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {task: head(embedding).squeeze(-1) for task, head in self.heads.items()}

    @staticmethod
    def masked_loss(predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor],
                    masks: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, int]]:
        losses, counts = [], {}
        for task, prediction in predictions.items():
            mask = masks[task].bool() & torch.isfinite(targets[task])
            counts[task] = int(mask.sum())
            if mask.any():
                losses.append(((prediction[mask] - targets[task][mask]) ** 2).mean())
        if not losses:
            raise ValueError("No valid property labels in batch")
        return torch.stack(losses).mean(), counts


class UnifiedPolymerModel(nn.Module):
    def __init__(self, structure_encoder: nn.Module, hidden_size: int = 128,
                 modality_dropout: float = 0.1, fusion: str = "gated"):
        super().__init__()
        self.structure_encoder = structure_encoder
        structure_size = structure_encoder.output_size
        self.md_encoder = MaskedChainMDEncoder(hidden_size)
        self.auxiliary_md = nn.Linear(structure_size, len(MD_FEATURES))
        self.fusion = GatedLateFusion(structure_size, hidden_size, len(MD_FEATURES), modality_dropout, fusion)
        self.properties = MultiTaskPropertyHeads(structure_size)

    def forward(self, psmiles: str | list[str], mode: str = "structure_only",
                md_values: torch.Tensor | None = None, md_availability: torch.Tensor | None = None,
                md_qc: torch.Tensor | None = None) -> dict[str, object]:
        structure = self.structure_encoder.encode_structure(psmiles)
        auxiliary = self.auxiliary_md(structure)
        if mode == "structure_only":
            fused = structure
        elif mode == "chain_md_predicted":
            inferred = torch.ones_like(auxiliary, dtype=torch.bool)
            encoded = self.md_encoder(auxiliary, inferred)
            fused = self.fusion(structure, encoded, inferred, mode)
        elif mode == "chain_md_observed":
            if md_values is None or md_availability is None:
                raise ValueError("Observed-MD mode requires values and masks")
            observed = md_availability.bool() & torch.isfinite(md_values) & (md_qc.bool() if md_qc is not None else True)
            encoded = self.md_encoder(md_values, md_availability, md_qc)
            fused = self.fusion(structure, encoded, observed, mode)
        else:
            raise ValueError(mode)
        return {"predictions": self.properties(fused), "auxiliary_md": auxiliary, "embedding": fused}
