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
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-validation", action="store_true", help="skip valid set (trainer-specific behavior)")
    parser.add_argument("--key-eval", type=str, default=None, help="override args['KeyEval'] (e.g. Loss, COPA_overall_acc)")

    # splitting
    parser.add_argument("--kfold-all", action="store_true", help="do KFold split on ORIGINAL train only (no original test)")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=1111)
    parser.add_argument("--only-fold", type=int, default=None, help="only run one fold (1..n_splits)")

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

    base_args = get_config_regression(model_name, dataset_name)
    base_args["device"] = device
    base_args["train_mode"] = "regression"
    if args_cli.skip_validation:
        base_args["skip_validation"] = True
    if args_cli.key_eval is not None:
        base_args["KeyEval"] = str(args_cli.key_eval)
    if args_cli.feature_path:
        base_args["featurePath"] = args_cli.feature_path
    if args_cli.max_epochs is not None:
        base_args["max_epochs"] = int(args_cli.max_epochs)
    if args_cli.batch_size is not None:
        base_args["batch_size"] = int(args_cli.batch_size)

    num_workers = args_cli.num_workers if args_cli.num_workers is not None else (8 if gpu_ids else 2)

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
        if dataset_name not in ["copa_1231", "custom", "train_12_16"]:
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
                pin_memory=bool(gpu_ids),
                persistent_workers=bool(gpu_ids),
                prefetch_factor=2,
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

        original_test_loader = _build_test_loader_only(base_args, num_workers=num_workers, pin_memory=bool(gpu_ids))

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

