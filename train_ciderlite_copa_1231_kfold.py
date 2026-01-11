#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIDerLite | copa_1231 | 5-Fold KFold 训练 + 测试（对齐 TFN 指标与输出风格）

重要说明：
- copa_valid_1231_converted.pkl 与 copa_test_1231_converted.pkl 在当前数据目录下重复；
  因此“全量数据”= train + test 两份 converted 数据合并后做 5-fold。
"""

import sys
import argparse
import copy
from pathlib import Path

# make import work no matter where you run this script from
project_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir / "src"))

import torch

from MMSA.config import get_config_regression
from MMSA.data_loader_kfold import COPA1231_KFoldDataLoader
from MMSA.models import AMIO
from MMSA.trains import ATIO
from MMSA.utils import setup_seed


def main():
    model_name = "cider_lite"
    dataset_name = "copa_1231"
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default=None, help="覆盖 args['featurePath']（用于 small 冒烟测试等）")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=1111)
    parser.add_argument("--only-fold", type=int, default=None, help="只跑某一个 fold（1..n_splits），用于快速确认可运行")
    parser.add_argument("--max-epochs", type=int, default=None, help="覆盖 max_epochs（用于冒烟测试）")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 batch_size（用于冒烟测试）")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader num_workers（GPU建议8~16）")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args_cli = parser.parse_args()

    n_splits = args_cli.n_splits
    split_seed = args_cli.split_seed

    gpu_ids = [] if args_cli.cpu else ([0] if torch.cuda.is_available() else [])
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")

    project_dir = Path(__file__).parent.absolute()

    print("=" * 70)
    print("CIDerLite | copa_1231 | 5-Fold KFold 训练 + 测试")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"KFold: n_splits={n_splits}, split_seed={split_seed}")

    base_args = get_config_regression(model_name, dataset_name)
    base_args["device"] = device
    base_args["train_mode"] = "regression"
    # 跳过验证集：每折仅训练并在该折 test 子集上评估
    base_args["skip_validation"] = True
    if args_cli.feature_path:
        base_args["featurePath"] = args_cli.feature_path
    if args_cli.max_epochs is not None:
        base_args["max_epochs"] = int(args_cli.max_epochs)
    if args_cli.batch_size is not None:
        base_args["batch_size"] = int(args_cli.batch_size)

    num_workers = args_cli.num_workers if args_cli.num_workers is not None else (8 if gpu_ids else 2)

    all_results = []
    fold_range = [args_cli.only_fold] if args_cli.only_fold else list(range(1, n_splits + 1))
    for fold_idx in fold_range:
        print(f"\n{'-'*70}")
        print(f"Fold {fold_idx}/{n_splits}")
        print(f"{'-'*70}")

        # keep EasyDict type (AMIO/ATIO use both args['x'] and args.x in different places)
        args = copy.deepcopy(base_args)
        args["cur_seed"] = fold_idx
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

        # scheme 1: set class-balanced weights into args for trainer
        args["pos_weight"] = split.pos_weight
        args["neg_weight"] = split.neg_weight

        print(f"train_size={len(dataloaders['train'].dataset)} | test_size={len(dataloaders['test'].dataset)}")
        print(f"train_pos={split.train_pos} train_neg={split.train_neg} pos_w={split.pos_weight:.4f} neg_w={split.neg_weight:.4f}")
        print(f"model_save_path={args['model_save_path']}")

        model = AMIO(args).to(args["device"])
        trainer = ATIO().getTrain(args)
        # trainer 使用 args.skip_validation 判断分支
        if hasattr(trainer, "args"):
            trainer.args.skip_validation = True

        trainer.do_train(model, dataloaders)

        # load best (or last) weights and evaluate on this fold's test subset
        model.load_state_dict(torch.load(args["model_save_path"], map_location=args["device"]))
        model.to(args["device"])
        test_results = trainer.do_test(model, dataloaders["test"], mode="TEST")
        all_results.append(test_results)

        print("\n📊 关键指标:")
        print(f"  COPA_overall_acc: {test_results.get('COPA_overall_acc', 0)*100:.2f}%")
        print(f"  Has0_acc_2: {test_results.get('Has0_acc_2', 0)*100:.2f}%")
        print(f"  Non0_acc_2: {test_results.get('Non0_acc_2', 0)*100:.2f}%")
        print(f"  Mult_acc_5: {test_results.get('Mult_acc_5', 0)*100:.2f}%")
        print(f"  Mult_acc_7: {test_results.get('Mult_acc_7', 0)*100:.2f}%")
        print(f"  MAE: {test_results.get('MAE', 0):.4f}")
        print(f"  Corr: {test_results.get('Corr', 0)*100:.2f}%")
        print(f"  Loss: {test_results.get('Loss', 0):.4f}")

    if all_results:
        print("\n" + "=" * 70)
        print("5-Fold 平均结果:")
        print("=" * 70)
        keys = all_results[0].keys()
        for key in keys:
            vals = [r.get(key, 0) for r in all_results]
            mean_val = sum(vals) / len(vals)
            if "acc" in key.lower() or key in ["Has0_acc_2", "Non0_acc_2", "Mult_acc_5", "Mult_acc_7", "Corr"]:
                print(f"  {key}: {mean_val*100:.2f}%")
            else:
                print(f"  {key}: {mean_val:.4f}")

    print("\n" + "=" * 70)
    print("KFold 训练测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

