from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from metrics import binary_average_precision, binary_roc_auc, format_metric, has_both_binary_classes
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler
from transformers import AutoConfig, AutoTokenizer, EsmModel

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
LABEL_MAPPING = {"LOF": 0, "GOF": 1}
CACHE_VERSION = 3


@dataclass(frozen=True)
class VariantRecord:
    wt_seq: str
    mut_seq: str
    label: int
    position: int | None = None


@dataclass(frozen=True)
class ModelCacheInfo:
    feature_dim: int
    hidden_size: int
    model_type: str | None
    resolved_commit_hash: str | None


def make_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def encode_sequence(tokenizer: Any, sequence: str, max_len: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        sequence,
        padding=False,
        truncation=True,
        max_length=max_len,
    )
    return {
        "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
    }


def infer_residue_token_offset(tokenizer: Any, max_len: int) -> int:
    try:
        with_special = tokenizer("A", padding=False, truncation=True, max_length=max_len)["input_ids"]
        without_special = tokenizer("A", add_special_tokens=False)["input_ids"]
    except (KeyError, TypeError):
        return 1

    if not without_special:
        return 1

    for offset in range(len(with_special) - len(without_special) + 1):
        if with_special[offset : offset + len(without_special)] == without_special:
            return offset

    raise ValueError("Could not locate residue tokens in tokenizer output; check tokenizer special-token policy")


def residue_position_to_token_index(position: int, residue_token_offset: int) -> int:
    return position + residue_token_offset - 1


class VariantDataset(Dataset):
    def __init__(self, records: list[VariantRecord], tokenizer: Any, max_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.residue_token_offset = infer_residue_token_offset(tokenizer, max_len)
        self._cache: list[dict[str, torch.Tensor] | None] = [None] * len(records)

    def __len__(self) -> int:
        return len(self.records)

    def _ensure_position_available(
        self,
        record: VariantRecord,
        encoded: dict[str, torch.Tensor],
        sequence_name: str,
    ) -> None:
        if record.position is None:
            return

        token_index = residue_position_to_token_index(record.position, self.residue_token_offset)
        if token_index >= encoded["input_ids"].numel() or encoded["attention_mask"][token_index].item() == 0:
            raise ValueError(
                f"position {record.position} for {sequence_name} sequence was truncated by max_len={self.max_len}; "
                "increase --max_len or train on sequence windows around the variant"
            )

    def _ensure_position_matches_token_difference(
        self,
        record: VariantRecord,
        wt_encoded: dict[str, torch.Tensor],
        mut_encoded: dict[str, torch.Tensor],
    ) -> None:
        if record.position is None:
            return

        token_index = residue_position_to_token_index(record.position, self.residue_token_offset)
        if wt_encoded["input_ids"][token_index].item() == mut_encoded["input_ids"][token_index].item():
            raise ValueError(
                f"position {record.position} does not identify a token difference between wild-type and mutant "
                "sequences; check that the CSV uses 1-based residue positions"
            )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cached = self._cache[idx]
        if cached is not None:
            return cached

        record = self.records[idx]

        wt_encoded = encode_sequence(self.tokenizer, record.wt_seq, self.max_len)
        mut_encoded = encode_sequence(self.tokenizer, record.mut_seq, self.max_len)
        self._ensure_position_available(record, wt_encoded, "wild-type")
        self._ensure_position_available(record, mut_encoded, "mutant")
        self._ensure_position_matches_token_difference(record, wt_encoded, mut_encoded)
        token_position = (
            residue_position_to_token_index(record.position, self.residue_token_offset)
            if record.position is not None
            else -1
        )

        item = {
            "wt_input_ids": wt_encoded["input_ids"],
            "wt_attention_mask": wt_encoded["attention_mask"],
            "mut_input_ids": mut_encoded["input_ids"],
            "mut_attention_mask": mut_encoded["attention_mask"],
            "label": torch.tensor(record.label, dtype=torch.long),
            "position": torch.tensor(token_position, dtype=torch.long),
        }
        self._cache[idx] = item
        return item


@dataclass(frozen=True)
class VariantCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return collate_variant_batch(batch, self.pad_token_id)


def collate_variant_batch(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    return {
        "wt_input_ids": nn.utils.rnn.pad_sequence(
            [item["wt_input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "wt_attention_mask": nn.utils.rnn.pad_sequence(
            [item["wt_attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        ),
        "mut_input_ids": nn.utils.rnn.pad_sequence(
            [item["mut_input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "mut_attention_mask": nn.utils.rnn.pad_sequence(
            [item["mut_attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        ),
        "label": torch.stack([item["label"] for item in batch]),
        "position": torch.stack([item["position"] for item in batch]),
    }


class CachedFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        if features.dim() != 2:
            raise ValueError("Cached features must be a 2D tensor")
        if labels.dim() != 1:
            raise ValueError("Cached labels must be a 1D tensor")
        if features.size(0) != labels.size(0):
            raise ValueError("Cached features and labels must contain the same number of examples")
        self.features = features.float().contiguous()
        self.labels = labels.long().contiguous()

    def __len__(self) -> int:
        return int(self.labels.size(0))

    @property
    def feature_dim(self) -> int:
        return int(self.features.size(1))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
        }


def pool_variant_embeddings(hidden_state: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
    cls_embedding = hidden_state[:, 0, :]
    if positions is None:
        return cls_embedding

    if positions.dim() != 1 or positions.size(0) != hidden_state.size(0):
        raise ValueError("positions must be a 1D tensor with one entry per sequence")

    positions = positions.to(hidden_state.device)
    valid_positions = positions >= 0
    if not bool(valid_positions.any()):
        return cls_embedding

    if bool((positions[valid_positions] >= hidden_state.size(1)).any()):
        raise ValueError("position index exceeds the tokenized sequence length")

    pooled = cls_embedding.clone()
    batch_indices = torch.arange(hidden_state.size(0), device=hidden_state.device)
    pooled[valid_positions] = hidden_state[
        batch_indices[valid_positions],
        positions[valid_positions],
        :,
    ]
    return pooled


def build_classifier(feature_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feature_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2),
    )


class FeatureOnlyClassifier(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.classifier = build_classifier(feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class ESM2VariantClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        freeze_backbone: bool = True,
        model_revision: str | None = None,
    ):
        super().__init__()
        self.esm = EsmModel.from_pretrained(
            model_name,
            revision=model_revision,
            add_pooling_layer=False,
        )
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False
            self.esm.eval()

        hidden_size = self.esm.config.hidden_size
        self.classifier = build_classifier(hidden_size * 3)

    def train(self, mode: bool = True) -> ESM2VariantClassifier:
        super().train(mode)
        if self.freeze_backbone:
            self.esm.eval()
        return self

    def encode_features(
        self,
        wt_input_ids: torch.Tensor,
        wt_attention_mask: torch.Tensor,
        mut_input_ids: torch.Tensor,
        mut_attention_mask: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grad_context = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with grad_context:
            wt_outputs = self.esm(input_ids=wt_input_ids, attention_mask=wt_attention_mask)
            mut_outputs = self.esm(input_ids=mut_input_ids, attention_mask=mut_attention_mask)

        wt_embedding = pool_variant_embeddings(wt_outputs.last_hidden_state, positions)
        mut_embedding = pool_variant_embeddings(mut_outputs.last_hidden_state, positions)

        diff = mut_embedding - wt_embedding
        return torch.cat((wt_embedding, mut_embedding, diff), dim=1)

    def forward(
        self,
        wt_input_ids: torch.Tensor | None = None,
        wt_attention_mask: torch.Tensor | None = None,
        mut_input_ids: torch.Tensor | None = None,
        mut_attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is None:
            if wt_input_ids is None or wt_attention_mask is None or mut_input_ids is None or mut_attention_mask is None:
                raise ValueError("Sequence inputs are required when features are not provided")
            features = self.encode_features(
                wt_input_ids,
                wt_attention_mask,
                mut_input_ids,
                mut_attention_mask,
                positions,
            )
        return self.classifier(features)


def parse_binary_label(value: str, line_number: int) -> int:
    normalized = value.strip()
    try:
        label = int(normalized)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: label must be 0 or 1, got {value!r}") from exc

    if label not in (0, 1):
        raise ValueError(f"Line {line_number}: label must be 0 or 1, got {value!r}")
    return label


def parse_position(value: str, line_number: int, position_col: str) -> int | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        position = int(normalized)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: {position_col} must be a 1-based integer, got {value!r}") from exc

    if position < 1:
        raise ValueError(f"Line {line_number}: {position_col} must be a 1-based integer, got {value!r}")
    return position


def validate_substitution_position(
    wt_seq: str,
    mut_seq: str,
    position: int | None,
    line_number: int,
    position_col: str,
) -> None:
    if position is None:
        return
    if len(wt_seq) != len(mut_seq):
        raise ValueError(
            f"Line {line_number}: {position_col} enables substitution-only position pooling, "
            f"but sequence lengths differ (wt={len(wt_seq)}, mut={len(mut_seq)})"
        )

    mismatches = [
        idx + 1
        for idx, (wt_residue, mut_residue) in enumerate(zip(wt_seq, mut_seq))
        if wt_residue != mut_residue
    ]
    if mismatches != [position]:
        raise ValueError(
            f"Line {line_number}: {position_col} enables substitution-only position pooling, "
            f"but mismatching residue positions are {mismatches or 'none'}"
        )


def load_variant_csv(
    path: str,
    wt_col: str,
    mut_col: str,
    label_col: str,
    position_col: str | None = None,
) -> list[VariantRecord]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {wt_col, mut_col, label_col}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        has_position_col = False
        if position_col:
            if position_col not in fieldnames:
                raise ValueError(f"{path} is missing required columns: {position_col}")
            has_position_col = True

        records: list[VariantRecord] = []
        for line_number, row in enumerate(reader, start=2):
            wt_seq = (row.get(wt_col) or "").strip()
            mut_seq = (row.get(mut_col) or "").strip()
            if not wt_seq:
                raise ValueError(f"Line {line_number}: {wt_col} must not be empty")
            if not mut_seq:
                raise ValueError(f"Line {line_number}: {mut_col} must not be empty")
            label = parse_binary_label(row.get(label_col) or "", line_number)
            position = (
                parse_position(row.get(position_col) or "", line_number, position_col)
                if has_position_col
                else None
            )
            if position is not None and (position > len(wt_seq) or position > len(mut_seq)):
                raise ValueError(
                    f"Line {line_number}: {position_col}={position} is outside the sequence length "
                    f"(wt={len(wt_seq)}, mut={len(mut_seq)})"
                )
            validate_substitution_position(wt_seq, mut_seq, position, line_number, position_col or "position")
            records.append(VariantRecord(wt_seq=wt_seq, mut_seq=mut_seq, label=label, position=position))

    if not records:
        raise ValueError(f"{path} contains no variant rows")
    return records


def make_mock_records() -> tuple[list[VariantRecord], list[VariantRecord]]:
    base = "MKTAYIAKQRQISFVKSHFSRQDILD"
    lof_mut = "MKTAYIAKQRQISFVKSHFSRQDILG"
    gof_mut = "MKTAYIAKQRQISFVKSHFSRQDIKD"
    train = [VariantRecord(base, lof_mut, 0) for _ in range(8)]
    train.extend(VariantRecord(base, gof_mut, 1) for _ in range(4))
    val = [VariantRecord(base, lof_mut, 0), VariantRecord(base, gof_mut, 1)]
    return train, val


def load_records(args: argparse.Namespace) -> tuple[list[VariantRecord], list[VariantRecord] | None]:
    if args.use_mock_data:
        return make_mock_records()
    if not args.train_csv:
        raise ValueError("--train_csv is required unless --use_mock_data is set")

    train_records = load_variant_csv(args.train_csv, args.wt_col, args.mut_col, args.label_col, args.position_col)
    val_records = None
    if args.val_csv:
        val_records = load_variant_csv(args.val_csv, args.wt_col, args.mut_col, args.label_col, args.position_col)
    return train_records, val_records


def compute_class_weights(
    records: list[VariantRecord],
    mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    if mode == "none":
        return None

    counts = Counter(record.label for record in records)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError("--class_weight auto requires both label classes in the training data")

    total = counts[0] + counts[1]
    weights = [total / (2 * counts[0]), total / (2 * counts[1])]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(worker_seed)


def is_main_process(is_distributed: bool) -> bool:
    return not is_distributed or dist.get_rank() == 0


def setup_device(args: argparse.Namespace) -> tuple[torch.device, bool, int]:
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    is_distributed = args.local_rank != -1
    if not is_distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Local training on {device}.")
        return device, False, -1

    if args.backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL distributed training requires CUDA. Use --backend gloo for CPU.")

    dist.init_process_group(args.backend)
    local_rank = args.local_rank
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    print(f"[Rank {local_rank}] Distributed training initialized with backend {args.backend}.")
    return device, True, local_rank


def build_dataloader(
    records: list[VariantRecord],
    tokenizer: Any,
    max_len: int,
    batch_size: int,
    num_workers: int,
    sampler: Any,
    seed: int | None = None,
) -> DataLoader:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for dynamic padding")

    dataset = VariantDataset(records, tokenizer, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=VariantCollator(pad_token_id),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed),
    )


def build_feature_dataloader(
    dataset: CachedFeatureDataset,
    batch_size: int,
    num_workers: int,
    sampler: Any,
    seed: int | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed),
    )


def hash_variant_records(records: list[VariantRecord]) -> str:
    payload = [
        {
            "wt_seq": record.wt_seq,
            "mut_seq": record.mut_seq,
            "label": record.label,
            "position": record.position,
        }
        for record in records
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_cache_metadata(
    records: list[VariantRecord],
    args: argparse.Namespace,
    split: str,
    model_info: ModelCacheInfo,
) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "split": split,
        "model_name": args.model_name,
        "model_revision": getattr(args, "model_revision", None),
        "model": {
            "name": args.model_name,
            "revision": getattr(args, "model_revision", None),
            "resolved_commit_hash": model_info.resolved_commit_hash,
            "hidden_size": model_info.hidden_size,
            "model_type": model_info.model_type,
        },
        "tokenizer": {
            "name": args.model_name,
            "revision": getattr(args, "model_revision", None),
            "resolved_commit_hash": model_info.resolved_commit_hash,
        },
        "max_len": args.max_len,
        "truncation": {
            "max_len": args.max_len,
            "tokenizer_truncation": True,
            "padding": "dynamic-batch",
            "position_index": "1-based-residue-position-plus-tokenizer-prefix-offset",
        },
        "feature_dim": model_info.feature_dim,
        "num_records": len(records),
        "records_hash": hash_variant_records(records),
        "pooling": "substitution-position-when-present-else-cls",
    }


def model_cache_info(model_name: str, model_revision: str | None = None) -> ModelCacheInfo:
    config = AutoConfig.from_pretrained(model_name, revision=model_revision)
    hidden_size = int(config.hidden_size)
    return ModelCacheInfo(
        feature_dim=hidden_size * 3,
        hidden_size=hidden_size,
        model_type=getattr(config, "model_type", None),
        resolved_commit_hash=getattr(config, "_commit_hash", None),
    )


def feature_dim_for_model(model_name: str, model_revision: str | None = None) -> int:
    return model_cache_info(model_name, model_revision).feature_dim


def metadata_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def load_cached_feature_dataset(cache_path: Path, expected_metadata: dict[str, Any]) -> CachedFeatureDataset:
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding cache payload must be a dict: {cache_path}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not metadata_matches(metadata, expected_metadata):
        raise ValueError(f"Embedding cache metadata does not match current run: {cache_path}")

    dataset = CachedFeatureDataset(payload["features"], payload["labels"])
    if dataset.feature_dim != expected_metadata["feature_dim"]:
        raise ValueError(f"Embedding cache feature dimension mismatch: {cache_path}")
    if len(dataset) != expected_metadata["num_records"]:
        raise ValueError(f"Embedding cache row count mismatch: {cache_path}")
    return dataset


def try_load_cached_feature_dataset(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> CachedFeatureDataset | None:
    if not cache_path.exists():
        return None
    try:
        return load_cached_feature_dataset(cache_path, expected_metadata)
    except (KeyError, RuntimeError, ValueError):
        return None


def save_feature_cache(
    cache_path: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(
        {
            "features": features.cpu().float().contiguous(),
            "labels": labels.cpu().long().contiguous(),
            "metadata": metadata,
        },
        tmp_path,
    )
    os.replace(tmp_path, cache_path)


def sequence_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["wt_input_ids"].to(device),
        batch["wt_attention_mask"].to(device),
        batch["mut_input_ids"].to(device),
        batch["mut_attention_mask"].to(device),
        batch["position"].to(device),
        batch["label"].to(device),
    )


def batch_outputs_and_labels(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "features" in batch:
        labels = batch["label"].to(device)
        outputs = model(features=batch["features"].to(device))
        return outputs, labels

    wt_ids, wt_mask, mut_ids, mut_mask, positions, labels = sequence_batch_to_device(batch, device)
    outputs = model(wt_ids, wt_mask, mut_ids, mut_mask, positions)
    return outputs, labels


def precompute_feature_cache(
    model: ESM2VariantClassifier,
    dataloader: DataLoader,
    device: torch.device,
    cache_path: Path,
    metadata: dict[str, Any],
) -> CachedFeatureDataset:
    cached_dataset = try_load_cached_feature_dataset(cache_path, metadata)
    if cached_dataset is not None:
        print(f"Reusing embedding cache: {cache_path}")
        return cached_dataset

    print(f"Building embedding cache: {cache_path}")
    model.eval()
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            wt_ids, wt_mask, mut_ids, mut_mask, positions, labels = sequence_batch_to_device(batch, device)
            features = model.encode_features(wt_ids, wt_mask, mut_ids, mut_mask, positions)
            feature_batches.append(features.cpu())
            label_batches.append(labels.cpu())

    features_all = torch.cat(feature_batches, dim=0)
    labels_all = torch.cat(label_batches, dim=0)
    save_feature_cache(cache_path, features_all, labels_all, metadata)
    return CachedFeatureDataset(features_all, labels_all)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_distributed: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        outputs, labels = batch_outputs_and_labels(model, batch, device)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    if is_distributed:
        totals = torch.tensor([total_loss, total_examples], dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total_examples = totals.tolist()

    return float(total_loss / max(total_examples, 1))


def evaluate_predictions(
    labels: list[int],
    positive_scores: list[float],
    pred_labels: list[int] | None = None,
    loss: float | None = None,
) -> dict[str, float | None]:
    if pred_labels is None:
        pred_labels = [1 if score >= 0.5 else 0 for score in positive_scores]

    correct = sum(int(pred == label) for pred, label in zip(pred_labels, labels))
    metrics: dict[str, float | None] = {
        "loss": loss,
        "accuracy": correct / len(labels) if labels else None,
        "auroc": None,
        "auprc": None,
    }
    if labels and has_both_binary_classes(labels):
        metrics["auroc"] = binary_roc_auc(labels, positive_scores)
        metrics["auprc"] = binary_average_precision(labels, positive_scores)
    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | None]:
    model.eval()
    labels_all: list[int] = []
    scores_all: list[float] = []
    preds_all: list[int] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs, labels = batch_outputs_and_labels(model, batch, device)
            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            labels_all.extend(labels.cpu().tolist())
            scores_all.extend(probabilities[:, 1].cpu().tolist())
            preds_all.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(total_examples, 1)
    return evaluate_predictions(labels_all, scores_all, preds_all, avg_loss)


def metrics_for_log(metrics: dict[str, float | None]) -> str:
    parts = []
    for key in ("loss", "accuracy", "auroc", "auprc"):
        parts.append(f"{key}={format_metric(metrics.get(key))}")
    return ", ".join(parts)


def model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    raw_model = unwrap_model(model)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state_dict(model),
            "checkpoint_model_type": raw_model.__class__.__name__,
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": args.model_name,
            "label_mapping": LABEL_MAPPING,
            "metrics": metrics,
            "config": vars(args),
        },
        path,
    )


def best_score(metrics: dict[str, float | None]) -> float:
    if metrics.get("auprc") is not None:
        return float(metrics["auprc"])
    if metrics.get("loss") is not None:
        return -float(metrics["loss"])
    return float("-inf")


def should_use_embedding_cache(args: argparse.Namespace) -> bool:
    if args.embedding_cache == "off":
        return False
    if args.embedding_cache == "on" and not args.freeze_backbone:
        raise ValueError("--embedding_cache on requires --freeze_backbone")
    return bool(args.freeze_backbone)


def resolve_embedding_cache_dir(args: argparse.Namespace) -> Path:
    if args.embedding_cache_dir:
        return Path(args.embedding_cache_dir)
    return Path(args.output_dir) / "embedding_cache"


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device, is_distributed, local_rank = setup_device(args)

    try:
        train_records, val_records = load_records(args)
        if is_main_process(is_distributed):
            print(f"Loaded {len(train_records)} training variants.")
            if val_records is not None:
                print(f"Loaded {len(val_records)} validation variants.")

        output_dir = Path(args.output_dir)
        use_embedding_cache = should_use_embedding_cache(args)
        embedding_cache_dir = resolve_embedding_cache_dir(args) if use_embedding_cache else None

        if is_main_process(is_distributed):
            output_dir.mkdir(parents=True, exist_ok=True)
            if embedding_cache_dir is not None:
                embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        if use_embedding_cache:
            model_info = model_cache_info(args.model_name, args.model_revision)
            feature_dim = model_info.feature_dim
            train_cache_path = embedding_cache_dir / "train.pt"
            val_cache_path = embedding_cache_dir / "val.pt" if val_records is not None else None
            train_cache_metadata = feature_cache_metadata(train_records, args, "train", model_info)
            val_cache_metadata = (
                feature_cache_metadata(val_records, args, "val", model_info) if val_records is not None else None
            )

            train_feature_dataset = None
            val_feature_dataset = None
            if is_main_process(is_distributed):
                train_feature_dataset = try_load_cached_feature_dataset(train_cache_path, train_cache_metadata)
                if val_records is not None and val_cache_path is not None and val_cache_metadata is not None:
                    val_feature_dataset = try_load_cached_feature_dataset(val_cache_path, val_cache_metadata)

                needs_precompute = train_feature_dataset is None or (
                    val_records is not None and val_feature_dataset is None
                )
                if needs_precompute:
                    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.model_revision)
                    encoder_model = ESM2VariantClassifier(
                        args.model_name,
                        freeze_backbone=True,
                        model_revision=args.model_revision,
                    ).to(device)
                    if train_feature_dataset is None:
                        train_precompute_loader = build_dataloader(
                            train_records,
                            tokenizer,
                            args.max_len,
                            args.batch_size,
                            args.num_workers,
                            SequentialSampler(train_records),
                            args.seed,
                        )
                        train_feature_dataset = precompute_feature_cache(
                            encoder_model,
                            train_precompute_loader,
                            device,
                            train_cache_path,
                            train_cache_metadata,
                        )
                    else:
                        print(f"Reusing embedding cache: {train_cache_path}")

                    if val_records is not None and val_cache_path is not None and val_cache_metadata is not None:
                        if val_feature_dataset is None:
                            val_precompute_loader = build_dataloader(
                                val_records,
                                tokenizer,
                                args.max_len,
                                args.batch_size,
                                args.num_workers,
                                SequentialSampler(val_records),
                                args.seed,
                            )
                            val_feature_dataset = precompute_feature_cache(
                                encoder_model,
                                val_precompute_loader,
                                device,
                                val_cache_path,
                                val_cache_metadata,
                            )
                        else:
                            print(f"Reusing embedding cache: {val_cache_path}")

                    del encoder_model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    print(f"Reusing embedding cache: {train_cache_path}")
                    if val_cache_path is not None:
                        print(f"Reusing embedding cache: {val_cache_path}")

            if is_distributed:
                dist.barrier()

            if train_feature_dataset is None:
                train_feature_dataset = load_cached_feature_dataset(train_cache_path, train_cache_metadata)
            if val_records is not None and val_cache_path is not None and val_cache_metadata is not None:
                if val_feature_dataset is None:
                    val_feature_dataset = load_cached_feature_dataset(val_cache_path, val_cache_metadata)

            train_sampler = (
                DistributedSampler(train_feature_dataset, seed=args.seed)
                if is_distributed
                else RandomSampler(train_feature_dataset, generator=make_generator(args.seed))
            )
            train_loader = build_feature_dataloader(
                train_feature_dataset,
                args.batch_size,
                args.num_workers,
                train_sampler,
                args.seed,
            )
            val_loader = None
            if val_feature_dataset is not None:
                val_loader = build_feature_dataloader(
                    val_feature_dataset,
                    args.batch_size,
                    args.num_workers,
                    SequentialSampler(val_feature_dataset),
                    args.seed,
                )
            model: nn.Module = FeatureOnlyClassifier(feature_dim).to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.model_revision)
            model = ESM2VariantClassifier(
                args.model_name,
                freeze_backbone=args.freeze_backbone,
                model_revision=args.model_revision,
            ).to(device)
            train_sampler = (
                DistributedSampler(train_records, seed=args.seed)
                if is_distributed
                else RandomSampler(train_records, generator=make_generator(args.seed))
            )
            train_loader = build_dataloader(
                train_records,
                tokenizer,
                args.max_len,
                args.batch_size,
                args.num_workers,
                train_sampler,
                args.seed,
            )
            val_loader = None
            if val_records is not None:
                val_loader = build_dataloader(
                    val_records,
                    tokenizer,
                    args.max_len,
                    args.batch_size,
                    args.num_workers,
                    SequentialSampler(val_records),
                    args.seed,
                )

        if is_distributed:
            if args.backend == "nccl":
                model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            else:
                model = DDP(model)

        class_weights = compute_class_weights(train_records, args.class_weight, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr,
        )

        if is_main_process(is_distributed):
            write_json(
                output_dir / "config.json",
                {
                    "args": vars(args),
                    "label_mapping": LABEL_MAPPING,
                    "train_size": len(train_records),
                    "val_size": len(val_records) if val_records is not None else 0,
                    "train_label_counts": dict(Counter(record.label for record in train_records)),
                    "embedding_cache": {
                        "enabled": use_embedding_cache,
                        "dir": str(embedding_cache_dir) if embedding_cache_dir is not None else None,
                    },
                },
            )

        history: list[dict[str, Any]] = []
        best_seen = float("-inf")
        for epoch in range(args.epochs):
            if is_distributed:
                train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, is_distributed)
            epoch_metrics: dict[str, Any] = {
                "epoch": epoch + 1,
                "train": {"loss": train_loss},
                "validation": None,
            }

            if is_main_process(is_distributed):
                log_line = f"Epoch {epoch + 1}/{args.epochs}, train_loss={train_loss:.4f}"
                if val_loader is not None:
                    val_metrics = evaluate(unwrap_model(model), val_loader, criterion, device)
                    epoch_metrics["validation"] = val_metrics
                    log_line += f", validation: {metrics_for_log(val_metrics)}"
                    if val_metrics["auroc"] is None or val_metrics["auprc"] is None:
                        log_line += " (AUROC/AUPRC skipped: validation labels need both classes)"
                    score = best_score(val_metrics)
                else:
                    score = -train_loss
                print(log_line)

                history.append(epoch_metrics)
                write_json(output_dir / "metrics.json", {"history": history})
                if score > best_seen:
                    best_seen = score
                    save_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, epoch + 1, args, epoch_metrics)
                save_checkpoint(output_dir / "last_checkpoint.pt", model, optimizer, epoch + 1, args, epoch_metrics)

            if is_distributed:
                dist.barrier()
    finally:
        if is_distributed:
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an ESM2 protein variant GOF/LOF classifier")

    parser.add_argument("--train_csv", type=str, help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, help="Path to validation CSV")
    parser.add_argument("--wt_col", type=str, default="wt_seq", help="Wild-type sequence column")
    parser.add_argument("--mut_col", type=str, default="mut_seq", help="Mutant sequence column")
    parser.add_argument("--label_col", type=str, default="label", help="Binary label column: 0=LOF, 1=GOF")
    parser.add_argument(
        "--position_col",
        type=str,
        help="Optional 1-based mutation position column; enables per-residue ESM embeddings when present",
    )
    parser.add_argument("--use_mock_data", action="store_true", help="Use tiny synthetic data for smoke tests")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/esm2_variant",
        help="Directory for checkpoints and metrics",
    )
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Hugging Face ESM model name")
    parser.add_argument("--model_revision", type=str, help="Optional Hugging Face model/tokenizer revision or commit")
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--embedding_cache",
        choices=["auto", "on", "off"],
        default="auto",
        help="Precompute and reuse frozen ESM features: auto uses cache when the backbone is frozen",
    )
    parser.add_argument(
        "--embedding_cache_dir",
        type=str,
        help="Directory for cached train.pt/val.pt features; defaults to output_dir/embedding_cache",
    )
    parser.add_argument("--class_weight", choices=["auto", "none"], default="auto")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--max_len", type=int, default=1024, help="Max tokenized sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")

    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend: nccl or gloo")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
