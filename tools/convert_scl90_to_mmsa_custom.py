#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert SCL90 list-of-samples pkl to MMSA custom pkl format.

Input (per file): a list[dict], each dict contains (observed from your files):
  - text: np.ndarray shape (768,)
  - video_feature: np.ndarray shape (197, 768)
  - audio_waveform: np.ndarray shape (L,)
  - ir_feature: np.ndarray shape (197, 768)
  - bio/eye/eeg/eda: variable-length arrays
  - label: float

Output (flat dict):
  - text: (N, 1, 768)
  - audio: (N, T, 3)  (simple frame stats from waveform; padded to max T)
  - vision: (N, 197, 768)
  - regression_labels: (N,)
  - audio_lengths, vision_lengths
  - extras (ir_feature/bio/eye/eeg/eda + *_lengths) if present
  - raw_text, id
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _frame_stats(wav: np.ndarray, frame_len: int, hop_len: int) -> Tuple[np.ndarray, int]:
    """
    Convert 1D waveform to (T, 3) features: mean, std, rms per frame.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        feats = np.zeros((1, 3), dtype=np.float32)
        return feats, 1

    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_len/hop_len must be > 0")

    T = max(1, 1 + int(math.floor((wav.size - frame_len) / hop_len))) if wav.size >= frame_len else 1
    feats = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        start = t * hop_len
        end = min(start + frame_len, wav.size)
        frame = wav[start:end]
        if frame.size == 0:
            continue
        feats[t, 0] = float(frame.mean())
        feats[t, 1] = float(frame.std())
        feats[t, 2] = float(np.sqrt(np.mean(frame * frame)))
    return feats, T


def _pad_3d(seqs: List[np.ndarray], pad_value: float = 0.0) -> Tuple[np.ndarray, List[int]]:
    """
    Pad a list of (Ti, D) arrays into (N, Tmax, D) and return lengths.
    """
    lengths = [int(s.shape[0]) for s in seqs]
    Tmax = max(lengths) if lengths else 0
    D = int(seqs[0].shape[1]) if seqs else 0
    out = np.full((len(seqs), Tmax, D), pad_value, dtype=np.float32)
    for i, s in enumerate(seqs):
        ti = s.shape[0]
        out[i, :ti, :] = s.astype(np.float32, copy=False)
    return out, lengths


def _pad_3d_from_var(seqs: List[np.ndarray], pad_value: float = 0.0) -> Tuple[np.ndarray, List[int]]:
    """
    Pad variable-length sequences (Ti, Di) but requiring same Di across samples.
    """
    if not seqs:
        return np.zeros((0, 0, 0), dtype=np.float32), []
    D = int(seqs[0].shape[1])
    for s in seqs:
        if s.ndim != 2 or int(s.shape[1]) != D:
            raise ValueError("All sequences must be 2D with the same feature dim")
    return _pad_3d(seqs, pad_value=pad_value)


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    """
    Ensure x is 2D:
    - (D,) -> (1, D)
    - scalar -> (1, 1)
    """
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim == 2:
        return x
    # flatten extra dims into feature dim
    return x.reshape(x.shape[0], -1)


def convert(in_path: Path, out_path: Path, frame_len: int, hop_len: int) -> None:
    with in_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty list in {in_path}, got {type(data)}")
    if not isinstance(data[0], dict):
        raise ValueError(f"Expected list[dict] in {in_path}, got list[{type(data[0])}]")

    N = len(data)
    # text -> (N, 1, 768)
    text = np.zeros((N, 1, 768), dtype=np.float32)
    vision = np.zeros((N, 197, 768), dtype=np.float32)
    ir_feature = np.zeros((N, 197, 768), dtype=np.float32)
    labels = np.zeros((N,), dtype=np.float32)
    ids = [f"{in_path.stem}_{i}" for i in range(N)]
    raw_text = [""] * N

    audio_seqs: List[np.ndarray] = []

    bio_seqs: List[np.ndarray] = []
    eye_seqs: List[np.ndarray] = []
    eeg_seqs: List[np.ndarray] = []
    eda_seqs: List[np.ndarray] = []

    has_bio = has_eye = has_eeg = has_eda = False
    has_ir = False

    for i, s in enumerate(data):
        # required-ish
        t = np.asarray(s["text"], dtype=np.float32).reshape(-1)
        if t.shape[0] != 768:
            raise ValueError(f"text dim != 768 at idx={i}, got {t.shape}")
        text[i, 0, :] = t

        v = np.asarray(s.get("video_feature", s.get("vision")), dtype=np.float32)
        if v.shape != (197, 768):
            raise ValueError(f"video_feature/vision shape != (197,768) at idx={i}, got {v.shape}")
        vision[i] = v

        if "ir_feature" in s:
            ir = np.asarray(s["ir_feature"], dtype=np.float32)
            if ir.shape != (197, 768):
                raise ValueError(f"ir_feature shape != (197,768) at idx={i}, got {ir.shape}")
            ir_feature[i] = ir
            has_ir = True

        labels[i] = float(s["label"])

        wav = np.asarray(s.get("audio_waveform", []), dtype=np.float32).reshape(-1)
        feats, _ = _frame_stats(wav, frame_len=frame_len, hop_len=hop_len)
        audio_seqs.append(feats)

        if "bio" in s:
            bio_seqs.append(_ensure_2d(np.asarray(s["bio"], dtype=np.float32)))
            has_bio = True
        if "eye" in s:
            eye_seqs.append(_ensure_2d(np.asarray(s["eye"], dtype=np.float32)))
            has_eye = True
        if "eeg" in s:
            eeg_seqs.append(_ensure_2d(np.asarray(s["eeg"], dtype=np.float32)))
            has_eeg = True
        if "eda" in s:
            eda_seqs.append(_ensure_2d(np.asarray(s["eda"], dtype=np.float32)))
            has_eda = True

    audio, audio_lengths = _pad_3d_from_var(audio_seqs, pad_value=0.0)
    vision_lengths = [197] * N

    out: Dict[str, object] = {
        "text": text,
        "audio": audio,
        "vision": vision,
        "regression_labels": labels,
        "audio_lengths": audio_lengths,
        "vision_lengths": vision_lengths,
        "raw_text": raw_text,
        "id": ids,
    }

    # extras
    if has_ir:
        out["ir_feature"] = ir_feature
        out["ir_feature_lengths"] = [197] * N
    if has_bio:
        bio, bio_lengths = _pad_3d_from_var(bio_seqs, pad_value=0.0)
        out["bio"] = bio
        out["bio_lengths"] = bio_lengths
    if has_eye:
        eye, eye_lengths = _pad_3d_from_var(eye_seqs, pad_value=0.0)
        out["eye"] = eye
        out["eye_lengths"] = eye_lengths
    if has_eeg:
        eeg, eeg_lengths = _pad_3d_from_var(eeg_seqs, pad_value=0.0)
        out["eeg"] = eeg
        out["eeg_lengths"] = eeg_lengths
    if has_eda:
        eda, eda_lengths = _pad_3d_from_var(eda_seqs, pad_value=0.0)
        out["eda"] = eda
        out["eda_lengths"] = eda_lengths

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"OK: {in_path} -> {out_path}")
    print(f"  N={N} text={text.shape} audio={audio.shape} vision={vision.shape} labels={labels.shape}")
    print(f"  audio_lengths: min={min(audio_lengths)} max={max(audio_lengths)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-train", type=str, required=True)
    ap.add_argument("--in-test", type=str, required=True)
    ap.add_argument("--out-train", type=str, required=True)
    ap.add_argument("--out-test", type=str, required=True)
    ap.add_argument("--frame-len", type=int, default=400)
    ap.add_argument("--hop-len", type=int, default=400)
    args = ap.parse_args()

    convert(Path(args.in_train), Path(args.out_train), frame_len=args.frame_len, hop_len=args.hop_len)
    convert(Path(args.in_test), Path(args.out_test), frame_len=args.frame_len, hop_len=args.hop_len)


if __name__ == "__main__":
    main()

