#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified training entry for MMSA.

Goals:
- Train any registered MMSA model via parameters: --model-name/--dataset-name
- Support two splitting modes:
  1) standard: use MMSA MMDataLoader (dataset's train/valid/test)
  2) kfold (COPA-like datasets): do KFold on ORIGINAL train only (original test is NOT used in training),
     then evaluate the saved fold models on ORIGINAL test.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from datetime import datetime

# make import work no matter where you run this script from
project_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir / "src"))

from MMSA.config import get_config_regression  # noqa: E402
from MMSA.data_loader import MMDataLoader, MMDataset  # noqa: E402
from MMSA.data_loader_kfold import COPA1231_KFoldDataLoader  # noqa: E402
from MMSA.models import AMIO  # noqa: E402
from MMSA.trains import ATIO  # noqa: E402
from MMSA.utils import setup_seed  # noqa: E402
from MMSA.utils.experiment import (  # noqa: E402
    append_rows_to_csv,
    configure_mmsa_logger,
    now_ts,
    save_json,
)


def _run_cider_full_from_cider_main(
    *,
    args_cli: argparse.Namespace,
    run_id: str,
    exp_tag: str,
    dataset_name: str,
    log_path: Path,
    results_csv: Path,
) -> None:
    """
    Run the *original* CIDer-main pipeline (full architecture, incl. BERT/counterfactual/recon)
    but integrate outputs into MMSA unified logs/csv.
    """
    cider_root = project_dir / "CIDer-main"
    if not cider_root.exists():
        raise FileNotFoundError(f"CIDer-main not found at: {cider_root}")

    # Make sure CIDer-main imports resolve (it uses top-level 'src' and 'modules')
    sys.path.insert(0, str(cider_root))

    import torch  # local import (safe)
    from torch.utils.data import DataLoader as TorchDataLoader  # noqa

    from bert_dataloader import MMDataset as CIDerDataset  # type: ignore
    from src import train_run  # type: ignore

    # device
    use_cuda = (not args_cli.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Build hyp_params Namespace compatible with CIDer-main
    # (we keep their defaults but allow overriding from our CLI)
    dataset = dataset_name.lower()
    task = str(getattr(args_cli, "cider_task", "binary")).lower()
    aligned = bool(getattr(args_cli, "cider_aligned", False))
    ood = bool(getattr(args_cli, "cider_ood", False))
    cross_dataset = bool(getattr(args_cli, "cider_cross_dataset", False))

    # data/model paths
    data_path = str(getattr(args_cli, "cider_data_path", "/root/autodl-tmp/data"))
    model_path = str(getattr(args_cli, "cider_model_path", str(project_dir / "saved_models" / "cider_full")))
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # training knobs
    batch_size = int(getattr(args_cli, "batch_size", None) or getattr(args_cli, "cider_batch_size", 128))
    lr = float(getattr(args_cli, "cider_lr", 1e-3))
    num_epochs = int(getattr(args_cli, "max_epochs", None) or getattr(args_cli, "cider_num_epochs", 200))
    patience = int(getattr(args_cli, "cider_patience", 10))
    seed = int(getattr(args_cli, "split_seed", 1111))

    # BERT knobs (critical: original repo hard-coded local paths, we override via our patched models.py)
    language = str(getattr(args_cli, "cider_language", "en"))
    bert_name_or_path = str(getattr(args_cli, "cider_bert_name_or_path", "bert-base-uncased" if language == "en" else "bert-base-chinese"))
    bert_cache_dir = getattr(args_cli, "cider_bert_cache_dir", None)

    # derive CIDer-main hyper params (copied from their main_run.py)
    output_dim_dict = {
        "mosi_binary": 3,
        "mosei_binary": 3,
        "mosi_seven": 7,
        "mosei_seven": 7,
        # copa_1231: labels are {-1,+1} -> 2-class {0,1}
        "copa_1231_binary": 2,
    }
    criterion_dict = {
        "mosi_binary": "NLLLoss",
        "mosei_binary": "NLLLoss",
        "mosi_seven": "NLLLoss",
        "mosei_seven": "NLLLoss",
        "copa_1231_binary": "NLLLoss",
    }

    hyp_params = argparse.Namespace()
    hyp_params.model = "CIDER"
    hyp_params.dataset = dataset
    hyp_params.task = task
    hyp_params.aligned = aligned
    hyp_params.ood = ood
    hyp_params.cross_dataset = cross_dataset
    hyp_params.data_path = data_path
    hyp_params.model_path = model_path
    hyp_params.batch_size = batch_size
    hyp_params.lr = lr
    hyp_params.optim = "Adam"
    hyp_params.num_epochs = num_epochs
    hyp_params.patience = patience
    hyp_params.bias_thresh = int(getattr(args_cli, "cider_bias_thresh", 100))
    hyp_params.seed = seed
    hyp_params.no_cuda = not use_cuda
    hyp_params.distribute = bool(getattr(args_cli, "cider_distribute", False))
    hyp_params.language = language
    hyp_params.missing_mode = str(getattr(args_cli, "cider_missing_mode", "RMFM"))
    hyp_params.missing_rate = [0.0]

    hyp_params.use_cuda = use_cuda
    hyp_params.weight_decay_bert = float(getattr(args_cli, "cider_weight_decay_bert", 1e-3))
    hyp_params.lr_bert = float(getattr(args_cli, "cider_lr_bert", 5e-5))
    hyp_params.bert_name_or_path = bert_name_or_path
    hyp_params.bert_cache_dir = bert_cache_dir

    # Fix feature dimensions and sequence length
    if dataset in ("mosi", "mosei"):
        if dataset == "mosi":
            hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 5, 20
            if aligned:
                hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 50, 50
            else:
                hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 375, 500
        else:
            hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 74, 35
            if aligned:
                hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 50, 50
            else:
                hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 500, 500
    elif dataset == "copa_1231":
        # Read actual dimensions from converted pkl to avoid hard-coding.
        import pickle  # local import
        from pathlib import Path as _Path  # local import
        train_p = _Path(data_path) / "copa_train_1231_converted.pkl"
        with open(str(train_p), "rb") as f:
            dd = pickle.load(f)
        dm = dd.get("train", list(dd.values())[0])
        # base
        audio = dm["audio"]
        vision = dm["vision"]
        d_a = int(audio.shape[2])
        d_v = int(vision.shape[2])
        if bool(getattr(args_cli, "use_all_features", False)):
            if "ir_feature" in dm and hasattr(dm["ir_feature"], "shape") and dm["ir_feature"].shape[:2] == vision.shape[:2]:
                d_v = int(vision.shape[2] + dm["ir_feature"].shape[2])
            # pooled dims: bio(3)+eye(2)+eeg(8)+eda(7) if present
            extra_d = 0
            for k, dim in [("bio", 3), ("eye", 2), ("eeg", 8), ("eda", 7)]:
                if k in dm:
                    extra_d += dim
            d_a = d_a + extra_d
        hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, d_a, d_v
        # sequence lengths from padded arrays
        hyp_params.l_len = int(dm["text_bert"].shape[2])
        hyp_params.a_len = int(audio.shape[1])
        hyp_params.v_len = int(vision.shape[1])
        # propagate flag to CIDer-main dataloader
        hyp_params.use_all_features = bool(getattr(args_cli, "use_all_features", False))
    else:
        raise ValueError(f"cider_full unsupported dataset={dataset}. Supported: mosi/mosei/copa_1231")

    hyp_params.output_dim = output_dim_dict.get(f"{dataset}_{task}", 1)
    hyp_params.criterion = criterion_dict.get(f"{dataset}_{task}", "NLLLoss")
    hyp_params.recon_loss = "SmoothL1Loss"

    # default hyperparams in main_run.py (keep upstream architecture behavior)
    hyp_params.embed_dim = int(getattr(args_cli, "cider_embed_dim", 32))
    hyp_params.multimodal_layers = int(getattr(args_cli, "cider_multimodal_layers", 3))
    hyp_params.num_heads = int(getattr(args_cli, "cider_num_heads", 8))
    hyp_params.attn_dropout = float(getattr(args_cli, "cider_attn_dropout", 0.0))
    hyp_params.out_dropout = float(getattr(args_cli, "cider_out_dropout", 0.3))
    hyp_params.embed_dropout = float(getattr(args_cli, "cider_embed_dropout", 0.2))
    hyp_params.relu_dropout = float(getattr(args_cli, "cider_relu_dropout", 0.1))
    hyp_params.res_dropout = float(getattr(args_cli, "cider_res_dropout", 0.1))
    hyp_params.joint_rep_weight = float(getattr(args_cli, "cider_joint_rep_weight", 0.5))
    hyp_params.attn_weight = float(getattr(args_cli, "cider_attn_weight", 0.1))
    hyp_params.recon_weight = float(getattr(args_cli, "cider_recon_weight", 0.6))

    # Use CIDer-main dataset/dataloader (full pipeline)
    train_data = CIDerDataset(hyp_params, "train")
    valid_data = CIDerDataset(hyp_params, "valid")
    test_data = CIDerDataset(hyp_params, "test")
    hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)

    nw = int(getattr(args_cli, "cider_num_workers", 4))
    train_loader = TorchDataLoader(train_data, batch_size=hyp_params.batch_size, shuffle=True, num_workers=nw, pin_memory=use_cuda, persistent_workers=(nw > 0))
    valid_loader = TorchDataLoader(valid_data, batch_size=hyp_params.batch_size, shuffle=False, num_workers=nw, pin_memory=use_cuda, persistent_workers=(nw > 0))
    test_loader = TorchDataLoader(test_data, batch_size=hyp_params.batch_size, shuffle=False, num_workers=nw, pin_memory=use_cuda, persistent_workers=(nw > 0))

    # run training
    metrics = train_run.initiate(hyp_params, train_loader, valid_loader, test_loader, device)
    if not isinstance(metrics, dict):
        metrics = {"best_value": metrics}

    # append to MMSA csv
    row: Dict[str, Any] = {
        "run_id": run_id,
        "script": "train_any.py",
        "model_name": "cider_full",
        "dataset_name": dataset,
        "split_mode": "standard",
        "exp_name": exp_tag,
        "ts": datetime.now().isoformat(),
        "device": str(device),
        "log_path": str(log_path),
        "model_path": str(model_path),
        "batch_size": hyp_params.batch_size,
        "max_epochs": hyp_params.num_epochs,
        "cider_task": task,
        "cider_aligned": aligned,
        "cider_ood": ood,
        "cider_cross_dataset": cross_dataset,
        "cider_bert_name_or_path": bert_name_or_path,
    }
    row.update(metrics)
    append_rows_to_csv(results_csv, [row])

    print("\nCIDer-full DONE. Metrics:")
    for k in sorted(metrics.keys()):
        print(f"  {k}: {metrics[k]}")
    print(f"APPEND CSV DONE: {results_csv}")


def _build_test_loader_only(args, num_workers: int, pin_memory: bool) -> DataLoader:
    """
    Build ONLY the original test dataloader without loading train/valid.
    This avoids MMDataLoader() side-effect of reading all splits.
    """
    test_ds = MMDataset(args, mode="test")
    return DataLoader(
        test_ds,
        batch_size=args["batch_size"],
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--smoke", action="store_true", help="fast sanity run: GPU(if available) + big batch + only-fold=1 + max-epochs=1")
    parser.add_argument("--exp-name", type=str, default=None, help="experiment tag to appear in log/csv filenames")
    parser.add_argument("--log-dir", type=str, default=None, help="directory to store python logs (default: mmsa/logs/experiments_py)")
    parser.add_argument("--results-csv", type=str, default=None, help="global csv path to append final test results")

    # common overrides
    parser.add_argument("--feature-path", type=str, default=None, help="override args['featurePath']")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Override text backbone (HF model id OR local dir). Example: /root/autodl-tmp/models/bert-base-uncased",
    )
    parser.add_argument(
        "--use-all-features",
        action="store_true",
        help="For COPA/custom datasets: fuse extra modalities (ir_feature/bio/eye/eeg/eda) into model inputs.",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-validation", action="store_true", help="skip valid set (trainer-specific behavior)")
    parser.add_argument("--key-eval", type=str, default=None, help="override args['KeyEval'] (e.g. Loss, COPA_overall_acc)")
    # dataloader knobs (primarily to avoid host OOM / Killed)
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch_factor when num_workers>0")
    pin_group = parser.add_mutually_exclusive_group()
    pin_group.add_argument("--pin-memory", action="store_true", help="force DataLoader pin_memory=True")
    pin_group.add_argument("--no-pin-memory", action="store_true", help="force DataLoader pin_memory=False")
    pw_group = parser.add_mutually_exclusive_group()
    pw_group.add_argument("--persistent-workers", action="store_true", help="force DataLoader persistent_workers=True (when num_workers>0)")
    pw_group.add_argument("--no-persistent-workers", action="store_true", help="force DataLoader persistent_workers=False")

    # splitting
    parser.add_argument("--kfold-all", action="store_true", help="do KFold split on ORIGINAL train only (no original test)")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=1111)
    parser.add_argument("--only-fold", type=int, default=None, help="only run one fold (1..n_splits)")

    # ===== CIDer-full (CIDer-main) integration =====
    parser.add_argument("--cider-task", type=str, default="binary", choices=["binary", "seven"], help="CIDer-main task")
    parser.add_argument("--cider-data-path", type=str, default="/root/autodl-tmp/data", help="root containing MOSI/MOSEI pkl files")
    parser.add_argument("--cider-model-path", type=str, default=None, help="where to save CIDer-main .pt weights")
    parser.add_argument("--cider-language", type=str, default="en", choices=["en", "cn"])
    parser.add_argument("--cider-bert-name-or-path", type=str, default=None, help="HF model id or local path (e.g. bert-base-uncased)")
    parser.add_argument("--cider-bert-cache-dir", type=str, default=None)
    parser.add_argument("--cider-num-workers", type=int, default=4)
    parser.add_argument("--cider-num-epochs", type=int, default=200)
    parser.add_argument("--cider-lr", type=float, default=1e-3)
    parser.add_argument("--cider-lr-bert", type=float, default=5e-5)
    parser.add_argument("--cider-weight-decay-bert", type=float, default=1e-3)
    parser.add_argument("--cider-patience", type=int, default=10)
    parser.add_argument("--cider-bias-thresh", type=int, default=100)
    parser.add_argument("--cider-aligned", action="store_true")
    parser.add_argument("--cider-ood", action="store_true")
    parser.add_argument("--cider-cross-dataset", action="store_true")
    parser.add_argument("--cider-distribute", action="store_true")
    parser.add_argument("--cider-missing-mode", type=str, default="RMFM")
    # optional hyper overrides
    parser.add_argument("--cider-embed-dim", type=int, default=32)
    parser.add_argument("--cider-multimodal-layers", type=int, default=3)
    parser.add_argument("--cider-num-heads", type=int, default=8)
    parser.add_argument("--cider-attn-dropout", type=float, default=0.0)
    parser.add_argument("--cider-out-dropout", type=float, default=0.3)
    parser.add_argument("--cider-embed-dropout", type=float, default=0.2)
    parser.add_argument("--cider-relu-dropout", type=float, default=0.1)
    parser.add_argument("--cider-res-dropout", type=float, default=0.1)
    parser.add_argument("--cider-joint-rep-weight", type=float, default=0.5)
    parser.add_argument("--cider-attn-weight", type=float, default=0.1)
    parser.add_argument("--cider-recon-weight", type=float, default=0.6)

    args_cli = parser.parse_args()

    model_name = args_cli.model_name.lower()
    dataset_name = args_cli.dataset_name.lower()
    split_mode = "kfold" if args_cli.kfold_all else "standard"

    # logging + result paths
    exp_tag = (args_cli.exp_name or "run").strip().replace(" ", "_")
    run_id = f"{model_name}__{dataset_name}__{split_mode}__{exp_tag}__{now_ts()}"
    log_dir = Path(args_cli.log_dir) if args_cli.log_dir else (project_dir / "logs" / "experiments_py")
    log_path = log_dir / f"{run_id}.log"
    logger = configure_mmsa_logger(log_path)
    results_csv = Path(args_cli.results_csv) if args_cli.results_csv else (project_dir / "results" / "all_final_test_results.csv")

    # apply smoke defaults (only if user didn't explicitly override)
    if args_cli.smoke:
        if args_cli.only_fold is None:
            args_cli.only_fold = 1
        if args_cli.max_epochs is None:
            args_cli.max_epochs = 1
        if args_cli.batch_size is None:
            # prefer a larger batch to speed up throughput on GPU; will fallback to CPU if no CUDA
            args_cli.batch_size = 512
        if args_cli.num_workers is None:
            args_cli.num_workers = 8

    gpu_ids = [] if args_cli.cpu else ([0] if torch.cuda.is_available() else [])
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")

    print("=" * 70)
    print(f"TRAIN | model={model_name} | dataset={dataset_name}")
    print("=" * 70)
    print(f"device: {device}")
    if args_cli.smoke:
        print(f"SMOKE: only_fold={args_cli.only_fold} max_epochs={args_cli.max_epochs} batch_size={args_cli.batch_size} num_workers={args_cli.num_workers}")
    print(f"log_path: {log_path}")
    print(f"results_csv: {results_csv}")

    logger.info(f"RUN_ID={run_id}")
    logger.info(f"model={model_name} dataset={dataset_name} split_mode={split_mode} device={device}")

    # Special-case: run original CIDer-main (full architecture) inside MMSA experiment infra.
    if model_name == "cider_full":
        _run_cider_full_from_cider_main(
            args_cli=args_cli,
            run_id=run_id,
            exp_tag=exp_tag,
            dataset_name=dataset_name,
            log_path=log_path,
            results_csv=results_csv,
        )
        return

    base_args = get_config_regression(model_name, dataset_name)
    base_args["device"] = device
    base_args["train_mode"] = "regression"
    if args_cli.skip_validation:
        base_args["skip_validation"] = True
    if args_cli.key_eval is not None:
        base_args["KeyEval"] = str(args_cli.key_eval)
    if args_cli.feature_path:
        base_args["featurePath"] = args_cli.feature_path
    if args_cli.pretrained:
        base_args["pretrained"] = str(args_cli.pretrained)
    # For custom/COPA-like datasets we can optionally fuse extra modalities into base modalities.
    # This keeps existing model architectures unchanged while actually consuming all available features.
    if args_cli.use_all_features:
        base_args["fuse_extra_features"] = True
    if args_cli.max_epochs is not None:
        base_args["max_epochs"] = int(args_cli.max_epochs)
    if args_cli.batch_size is not None:
        base_args["batch_size"] = int(args_cli.batch_size)

    num_workers = args_cli.num_workers if args_cli.num_workers is not None else (8 if gpu_ids else 2)
    prefetch_factor = int(args_cli.prefetch_factor)
    if prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    if args_cli.pin_memory:
        pin_memory = True
    elif args_cli.no_pin_memory:
        pin_memory = False
    else:
        pin_memory = bool(gpu_ids)
    if args_cli.persistent_workers:
        persistent_workers = True
    elif args_cli.no_persistent_workers:
        persistent_workers = False
    else:
        persistent_workers = bool(gpu_ids)

    # save run config
    save_json(
        project_dir / "results" / "runs" / f"{run_id}.json",
        {
            "run_id": run_id,
            "ts": datetime.now().isoformat(),
            "script": "train_any.py",
            "model_name": model_name,
            "dataset_name": dataset_name,
            "split_mode": split_mode,
            "device": str(device),
            "log_path": str(log_path),
            "results_csv": str(results_csv),
            "cli": vars(args_cli),
            "base_args": dict(base_args),
        },
    )

    # mode 1: kfold (COPA-like datasets)
    if args_cli.kfold_all:
        if dataset_name not in ["copa_1231", "custom", "train_12_16", "scl90_1231"]:
            raise ValueError("--kfold-all currently supported only for COPA-like datasets (copa_1231/custom/train_12_16).")
        n_splits = int(args_cli.n_splits)
        split_seed = int(args_cli.split_seed)
        fold_range = [args_cli.only_fold] if args_cli.only_fold else list(range(1, n_splits + 1))

        for fold_idx in fold_range:
            print(f"\n{'-'*70}")
            print(f"Fold {fold_idx}/{n_splits} (kfold)")
            print(f"{'-'*70}")
            logger.info(f"Fold {fold_idx}/{n_splits} START")

            args = copy.deepcopy(base_args)
            args["cur_seed"] = fold_idx
            # kfold 没有 valid 集，因此强制走 skip_validation 分支
            args["skip_validation"] = True
            setup_seed(1111 + fold_idx)

            fold_save_dir = project_dir / "saved_models" / f"fold{fold_idx}"
            fold_save_dir.mkdir(parents=True, exist_ok=True)
            args["model_save_path"] = fold_save_dir / f"{model_name}-{dataset_name}.pth"

            dataloaders, split = COPA1231_KFoldDataLoader(
                args,
                num_workers=num_workers,
                fold_idx=fold_idx,
                n_splits=n_splits,
                split_seed=split_seed,
                include_test_part=False,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            # optional: some trainers may use these
            args["pos_weight"] = split.pos_weight
            args["neg_weight"] = split.neg_weight

            print(f"train_size={len(dataloaders['train'].dataset)} | test_size={len(dataloaders['test'].dataset)}")
            print(f"model_save_path={args['model_save_path']}")
            logger.info(f"Fold {fold_idx}: train_size={len(dataloaders['train'].dataset)} test_size={len(dataloaders['test'].dataset)}")
            logger.info(f"Fold {fold_idx}: model_save_path={args['model_save_path']}")

            model = AMIO(args).to(args["device"])
            trainer = ATIO().getTrain(args)
            if hasattr(trainer, "args"):
                trainer.args.skip_validation = True
            trainer.do_train(model, dataloaders)
            print("done.")
            logger.info(f"Fold {fold_idx}/{n_splits} TRAIN DONE")

        # After KFold training, directly evaluate each fold model on ORIGINAL test once.
        print("\n" + "=" * 70)
        print("EVAL ON ORIGINAL TEST (no extra final training)")
        print("=" * 70)

        original_test_loader = _build_test_loader_only(base_args, num_workers=num_workers, pin_memory=pin_memory)

        all_test_results = []
        csv_rows = []
        for fold_idx in fold_range:
            args = copy.deepcopy(base_args)
            args["cur_seed"] = fold_idx
            fold_model_path = project_dir / "saved_models" / f"fold{fold_idx}" / f"{model_name}-{dataset_name}.pth"
            if not fold_model_path.exists():
                print(f"❌ missing weights: {fold_model_path}")
                continue
            model = AMIO(args).to(args["device"])
            model.load_state_dict(torch.load(fold_model_path, map_location=args["device"]))
            model.eval()
            trainer = ATIO().getTrain(args)
            res = trainer.do_test(model, original_test_loader, mode="TEST")
            all_test_results.append(res)
            logger.info(f"Fold {fold_idx} ORIGINAL TEST: {res}")
            print(
                f"Fold {fold_idx} ORIGINAL TEST >> "
                f"COPA_overall_acc={res.get('COPA_overall_acc', 0)*100:.2f}% "
                f"MAE={res.get('MAE', 0):.4f} "
                f"Corr={res.get('Corr', 0)*100:.2f}% "
                f"Loss={res.get('Loss', 0):.4f}"
            )

            csv_rows.append(
                {
                    "run_id": run_id,
                    "script": "train_any.py",
                    "split_mode": "kfold",
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "exp_name": exp_tag,
                    "ts": datetime.now().isoformat(),
                    "fold_idx": fold_idx,
                    "n_splits": n_splits,
                    "split_seed": split_seed,
                    "device": str(device),
                    "batch_size": int(base_args.get("batch_size", 0) or 0),
                    "max_epochs": int(base_args.get("max_epochs", 0) or 0),
                    "KeyEval": str(base_args.get("KeyEval", "")),
                    "featurePath": str(base_args.get("featurePath", "")),
                    "model_path": str(fold_model_path),
                    "log_path": str(log_path),
                    **res,
                }
            )

        if all_test_results:
            print("\n" + "=" * 70)
            print("ORIGINAL TEST (mean over folds):")
            print("=" * 70)
            keys = all_test_results[0].keys()
            mean_row = {
                "run_id": run_id,
                "script": "train_any.py",
                "split_mode": "kfold",
                "model_name": model_name,
                "dataset_name": dataset_name,
                "exp_name": exp_tag,
                "ts": datetime.now().isoformat(),
                "fold_idx": "mean",
                "n_splits": n_splits,
                "split_seed": split_seed,
                "device": str(device),
                "batch_size": int(base_args.get("batch_size", 0) or 0),
                "max_epochs": int(base_args.get("max_epochs", 0) or 0),
                "KeyEval": str(base_args.get("KeyEval", "")),
                "featurePath": str(base_args.get("featurePath", "")),
                "model_path": "",
                "log_path": str(log_path),
            }
            for key in keys:
                vals = [r.get(key, 0) for r in all_test_results]
                mean_val = sum(vals) / len(vals)
                mean_row[key] = mean_val
                if "acc" in key.lower() or key in ["Has0_acc_2", "Non0_acc_2", "Mult_acc_5", "Mult_acc_7", "Corr"]:
                    print(f"  {key}: {mean_val*100:.2f}%")
                else:
                    print(f"  {key}: {mean_val:.4f}")

            # save csv (final/original test only)
            append_rows_to_csv(results_csv, [*csv_rows, mean_row])
            logger.info(f"APPEND CSV DONE: {results_csv}")

        return

    # mode 2: standard split (MMDataLoader)
    print("\nloading standard dataloaders...")
    dataloader = MMDataLoader(base_args, num_workers=num_workers)
    print(f"train={len(dataloader['train'].dataset)} valid={len(dataloader['valid'].dataset)} test={len(dataloader['test'].dataset)}")

    # use 5 seeds as folds by convention
    seeds = [1111, 1112, 1113, 1114, 1115]
    fold_range = [args_cli.only_fold] if args_cli.only_fold else list(range(1, len(seeds) + 1))
    for idx in fold_range:
        seed = seeds[idx - 1]
        print(f"\n{'-'*70}")
        print(f"Fold {idx}/5 (seed={seed})")
        print(f"{'-'*70}")
        logger.info(f"Fold {idx}/5 (standard) START seed={seed}")

        args = copy.deepcopy(base_args)
        args["cur_seed"] = idx
        setup_seed(seed)

        fold_save_dir = project_dir / "saved_models" / f"fold{idx}"
        fold_save_dir.mkdir(parents=True, exist_ok=True)
        args["model_save_path"] = fold_save_dir / f"{model_name}-{dataset_name}.pth"
        print(f"model_save_path={args['model_save_path']}")

        model = AMIO(args).to(args["device"])
        trainer = ATIO().getTrain(args)
        if hasattr(trainer, "args"):
            trainer.args.skip_validation = bool(args.get("skip_validation", False))
        trainer.do_train(model, dataloader)
        print("done.")
        logger.info(f"Fold {idx}/5 (standard) TRAIN DONE")

        # final test for this fold (standard)
        model.load_state_dict(torch.load(args["model_save_path"], map_location=args["device"]))
        model.to(args["device"])
        model.eval()
        res = trainer.do_test(model, dataloader["test"], mode="TEST")
        logger.info(f"Fold {idx} TEST: {res}")
        append_rows_to_csv(
            results_csv,
            [
                {
                    "run_id": run_id,
                    "script": "train_any.py",
                    "split_mode": "standard",
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "exp_name": exp_tag,
                    "ts": datetime.now().isoformat(),
                    "fold_idx": idx,
                    "n_splits": 5,
                    "split_seed": "",
                    "seed": seed,
                    "device": str(device),
                    "batch_size": int(base_args.get("batch_size", 0) or 0),
                    "max_epochs": int(base_args.get("max_epochs", 0) or 0),
                    "KeyEval": str(base_args.get("KeyEval", "")),
                    "featurePath": str(base_args.get("featurePath", "")),
                    "model_path": str(args["model_save_path"]),
                    "log_path": str(log_path),
                    **res,
                }
            ],
        )


if __name__ == "__main__":
    main()

