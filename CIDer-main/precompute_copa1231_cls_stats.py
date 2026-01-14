#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute CIDer-main required cls_probs / cls_feats for copa_1231 (Scheme B).

CIDer-main expects:
- {dataset}_probs/{iid|ood}/{task}/{dataset}_train_{task}.npy
    shape: (C,) float32  # class prior
- {dataset}_feats/{iid|ood}/{task}/{dataset}_{mode}_{task}_{aligned|unaligned}_{idx}.npy
    shape: (D_l + D_a + D_v,) float32  # class prototype feature (text/audio/vision concat)

For copa_1231:
- labels are regression_labels in {-1, +1}. We map to 2-class: {-1 -> 0, +1 -> 1}.
- text prototype: BERT [CLS] embedding (last_hidden_state[:,0,:]) averaged per class.
- audio/vision prototype: masked mean pooling over time (lengths) averaged per class.
- If --use-all-features:
  - vision := concat(vision, ir_feature)  -> D_v=1536
  - audio  := concat(audio, pooled(bio/eye/eeg/eda) broadcast over time) -> D_a=13+20=33
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel


def _load_mode(path: Path, mode: str) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    if mode in data:
        return data[mode]
    if len(data.keys()) == 1:
        return list(data.values())[0]
    # copa_valid_1231_converted.pkl may store only "test"
    if mode == "valid" and "test" in data:
        return data["test"]
    raise KeyError(f"Cannot find mode={mode} in {path}. keys={list(data.keys())}")


def _masked_mean_np(x: np.ndarray, lengths: list[int] | np.ndarray) -> np.ndarray:
    """
    x: (N, L, D)
    lengths: (N,)
    returns: (N, D)
    """
    n, l, d = x.shape
    lens = np.asarray(lengths, dtype=np.int32)
    lens = np.clip(lens, 1, l)
    mask = (np.arange(l, dtype=np.int32)[None, :] < lens[:, None]).astype(np.float32)[:, :, None]
    denom = np.clip(mask.sum(axis=1), 1.0, None)
    return (x * mask).sum(axis=1) / denom


def _fuse_all_features(dm: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Returns:
      audio_fused: (N, La, Da')
      vision_fused:(N, Lv, Dv')
      Da', Dv'
    """
    audio = dm["audio"].astype(np.float32)
    vision = dm["vision"].astype(np.float32)

    # vision <- concat ir_feature if present and compatible
    if "ir_feature" in dm:
        ir = dm["ir_feature"].astype(np.float32)
        if ir.ndim == 3 and ir.shape[0] == vision.shape[0] and ir.shape[1] == vision.shape[1]:
            vision = np.concatenate([vision, ir], axis=2)

    # audio <- concat pooled bio/eye/eeg/eda (broadcast over time)
    pooled_list = []
    for k in ["bio", "eye", "eeg", "eda"]:
        if k in dm:
            arr = dm[k].astype(np.float32)
            if arr.ndim != 3:
                continue
            lens = dm.get(f"{k}_lengths", [arr.shape[1]] * arr.shape[0])
            pooled_list.append(_masked_mean_np(arr, lens))
    if pooled_list:
        pooled = np.concatenate(pooled_list, axis=1)  # (N, sumD)
        rep = np.repeat(pooled[:, None, :], audio.shape[1], axis=1).astype(np.float32)
        audio = np.concatenate([audio, rep], axis=2)

    return audio, vision, audio.shape[2], vision.shape[2]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="/root/autodl-tmp/data")
    ap.add_argument("--bert-name-or-path", type=str, required=True, help="HF model id or local dir")
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use-all-features", action="store_true")
    ap.add_argument("--out-root", type=str, default=None, help="CIDer-main root (default: this file's parent)")
    args = ap.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    cider_root = Path(args.out_root) if args.out_root else Path(__file__).resolve().parent

    train_path = data_dir / "copa_train_1231_converted.pkl"
    valid_path = data_dir / "copa_valid_1231_converted.pkl"
    test_path = data_dir / "copa_test_1231_converted.pkl"

    # Load train for priors + prototypes (we reuse same prototypes for valid/test feats files)
    train_dm = _load_mode(train_path, "train")

    # labels: {-1,+1} -> {0,1}
    y = train_dm["regression_labels"].astype(np.float32)
    y01 = (y > 0).astype(np.int64)
    cls_id = 2
    counts = np.bincount(y01, minlength=cls_id).astype(np.float32)
    priors = counts / np.clip(counts.sum(), 1.0, None)

    # Prepare features
    if args.use_all_features:
        audio_np, vision_np, d_a, d_v = _fuse_all_features(train_dm)
    else:
        audio_np = train_dm["audio"].astype(np.float32)
        vision_np = train_dm["vision"].astype(np.float32)
        d_a, d_v = audio_np.shape[2], vision_np.shape[2]

    audio_vec = _masked_mean_np(audio_np, train_dm.get("audio_lengths", [audio_np.shape[1]] * audio_np.shape[0]))
    vision_vec = _masked_mean_np(vision_np, train_dm.get("vision_lengths", [vision_np.shape[1]] * vision_np.shape[0]))

    text_bert = train_dm["text_bert"].astype(np.float32)  # (N,3,50)
    input_ids = torch.from_numpy(text_bert[:, 0, :]).long()
    attn_mask = torch.from_numpy(text_bert[:, 1, :]).long()
    token_type = torch.from_numpy(text_bert[:, 2, :]).long()

    bert = BertModel.from_pretrained(
        args.bert_name_or_path,
        cache_dir=args.cache_dir,
        local_files_only=(Path(args.bert_name_or_path).exists()),
    ).to(device)
    bert.eval()

    n = input_ids.shape[0]
    bs = int(args.batch_size)
    text_cls = np.zeros((n, 768), dtype=np.float32)
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        out = bert(
            input_ids=input_ids[sl].to(device),
            attention_mask=attn_mask[sl].to(device),
            token_type_ids=token_type[sl].to(device),
        ).last_hidden_state[:, 0, :]  # (B,768)
        text_cls[sl] = out.detach().cpu().numpy().astype(np.float32)

    # class prototypes
    d_l = 768
    proto = []
    for c in range(cls_id):
        m = (y01 == c)
        if m.sum() == 0:
            # avoid NaN
            proto_l = np.zeros((d_l,), dtype=np.float32)
            proto_a = np.zeros((d_a,), dtype=np.float32)
            proto_v = np.zeros((d_v,), dtype=np.float32)
        else:
            proto_l = text_cls[m].mean(axis=0).astype(np.float32)
            proto_a = audio_vec[m].mean(axis=0).astype(np.float32)
            proto_v = vision_vec[m].mean(axis=0).astype(np.float32)
        proto.append(np.concatenate([proto_l, proto_a, proto_v], axis=0).astype(np.float32))

    # write files
    dataset = "copa_1231"
    task = "binary"
    distribution = "iid"
    probs_dir = cider_root / f"{dataset}_probs" / distribution / task
    feats_dir = cider_root / f"{dataset}_feats" / distribution / task
    probs_dir.mkdir(parents=True, exist_ok=True)
    feats_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(probs_dir / f"{dataset}_train_{task}.npy"), priors.astype(np.float32))

    for mode in ["train", "valid", "test"]:
        for version in ["aligned", "unaligned"]:
            for idx in range(cls_id):
                np.save(str(feats_dir / f"{dataset}_{mode}_{task}_{version}_{idx}.npy"), proto[idx])

    print("DONE.")
    print(f"priors: {priors.tolist()}")
    print(f"proto_dim: {proto[0].shape[0]} (d_l={d_l}, d_a={d_a}, d_v={d_v})")
    print(f"wrote: {probs_dir / f'{dataset}_train_{task}.npy'}")
    print(f"wrote feats under: {feats_dir}")


if __name__ == "__main__":
    main()

