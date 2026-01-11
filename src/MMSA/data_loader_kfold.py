"""
K-Fold DataLoader utilities for COPA_1231.

说明：
- copa_valid_1231_converted.pkl 与 copa_test_1231_converted.pkl 在当前数据目录下完全一致（重复数据）。
- “使用所有数据”时，我们仅合并 train + test 两份 converted 数据作为全集，然后做 5-fold。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .data_loader import MMDataset


@dataclass(frozen=True)
class KFoldSplit:
    fold_idx: int
    n_splits: int
    seed: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_pos: int
    train_neg: int
    pos_weight: float
    neg_weight: float


def _pad_2d(batch_tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    # tensors: (L, D) with possibly different L, same D
    max_len = max(t.shape[0] for t in batch_tensors)
    feat_dim = batch_tensors[0].shape[1]
    out = batch_tensors[0].new_full((len(batch_tensors), max_len, feat_dim), float(pad_value))
    for i, t in enumerate(batch_tensors):
        out[i, : t.shape[0], :] = t
    return out


def _default_collate_with_padding(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    针对 COPA full(KFold) 场景：train 与 test 的序列长度可能不同（如 audio=752 vs 771，bio=25 vs 22）。
    默认 PyTorch collate 会因 shape 不一致而报错，这里对 (L,D) 的 Tensor 做 batch 内 padding。
    """
    assert batch, "empty batch"
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        v0 = vals[0]

        if isinstance(v0, dict):
            # labels dict
            sub = {}
            for sub_k in v0.keys():
                sub_vals = [v[sub_k] for v in vals]
                sub[sub_k] = torch.stack(sub_vals, dim=0)
            out[k] = sub
            continue

        if isinstance(v0, str):
            out[k] = vals
            continue

        if isinstance(v0, (int, float, np.integer, np.floating)):
            out[k] = torch.tensor(vals)
            continue

        if torch.is_tensor(v0):
            # common cases:
            # - text: (L, D) fixed
            # - audio/vision/extras: (L, D) variable L -> pad
            if all(torch.is_tensor(v) for v in vals):
                shapes = [tuple(v.shape) for v in vals]
                if all(s == shapes[0] for s in shapes):
                    out[k] = torch.stack(vals, dim=0)
                else:
                    # support (L, D)
                    if v0.dim() == 2 and all(v.dim() == 2 and v.shape[1] == v0.shape[1] for v in vals):
                        out[k] = _pad_2d(vals, pad_value=0.0)
                    else:
                        raise RuntimeError(f"Unsupported variable shape for key={k}: shapes={shapes}")
            continue

        # fallback: keep list
        out[k] = vals
    return out


def _kfold_indices(n: int, n_splits: int, seed: int, fold_idx: int) -> KFoldSplit:
    if n_splits <= 1:
        raise ValueError("n_splits must be >= 2")
    if not (1 <= fold_idx <= n_splits):
        raise ValueError(f"fold_idx must be in [1, {n_splits}]")

    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    test_indices = folds[fold_idx - 1]
    train_indices = np.concatenate([f for i, f in enumerate(folds) if i != (fold_idx - 1)], axis=0)

    # placeholders; will be filled by caller that has access to labels
    return KFoldSplit(
        fold_idx=fold_idx,
        n_splits=n_splits,
        seed=seed,
        train_indices=train_indices,
        test_indices=test_indices,
        train_pos=0,
        train_neg=0,
        pos_weight=1.0,
        neg_weight=1.0,
    )


def COPA1231_KFoldDataLoader(
    args,
    num_workers: int,
    fold_idx: int,
    n_splits: int = 5,
    split_seed: int = 1111,
    include_test_part: bool = False,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> Tuple[Dict[str, DataLoader], KFoldSplit]:
    """
    返回与 MMSA 训练器兼容的 dataloader dict：{'train':..., 'test':...}

    include_test_part=False 表示只在原始 train 上做 KFold（不触碰原始 test），更符合严格评估。
    """
    base_train = MMDataset(args, mode="train")
    base_test = None

    datasets: List[Dataset] = [base_train]
    if include_test_part:
        base_test = MMDataset(args, mode="test")
        datasets.append(base_test)

    full_ds = ConcatDataset(datasets)
    split_raw = _kfold_indices(len(full_ds), n_splits=n_splits, seed=split_seed, fold_idx=fold_idx)

    # compute class weights on TRAIN subset (binary view: y>=0 is positive)
    # y is expected to be {-1, +1} for copa_1231, but we keep threshold for safety.
    y_train = np.asarray(base_train.labels['M'], dtype=np.float32)
    y_test = np.asarray(base_test.labels['M'], dtype=np.float32) if (include_test_part and base_test is not None) else None
    n_train = len(base_train)
    pos = 0
    neg = 0
    for idx in split_raw.train_indices:
        if idx < n_train:
            y = y_train[int(idx)]
        else:
            if y_test is None:
                # should not happen
                continue
            y = y_test[int(idx - n_train)]
        if y >= 0:
            pos += 1
        else:
            neg += 1

    total = max(pos + neg, 1)
    # balanced re-weighting so each class contributes ~ equally
    pos_w = (total / (2 * pos)) if pos > 0 else 1.0
    neg_w = (total / (2 * neg)) if neg > 0 else 1.0
    split = KFoldSplit(
        fold_idx=split_raw.fold_idx,
        n_splits=split_raw.n_splits,
        seed=split_raw.seed,
        train_indices=split_raw.train_indices,
        test_indices=split_raw.test_indices,
        train_pos=pos,
        train_neg=neg,
        pos_weight=float(pos_w),
        neg_weight=float(neg_w),
    )

    train_ds = Subset(full_ds, split.train_indices.tolist())
    test_ds = Subset(full_ds, split.test_indices.tolist())

    # dataloader perf knobs
    if pin_memory is None:
        device = args.get("device", None) if isinstance(args, dict) else getattr(args, "device", None)
        pin_memory = bool(getattr(device, "type", "") == "cuda")
    if persistent_workers is None:
        persistent_workers = bool(num_workers and num_workers > 0)
    # prefetch_factor is only valid when num_workers > 0
    if prefetch_factor is None:
        prefetch_factor = 2
    dl_kwargs = {}
    if num_workers and num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)
        dl_kwargs["persistent_workers"] = bool(persistent_workers)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=args["batch_size"] if isinstance(args, dict) else args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_default_collate_with_padding,
            **dl_kwargs,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=args["batch_size"] if isinstance(args, dict) else args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_default_collate_with_padding,
            **dl_kwargs,
        ),
    }

    return loaders, split

