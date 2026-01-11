import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')


class CIDerLite():
    """
    Trainer for CIDerLite regression model.
    复用 TFN 的训练/测试逻辑与 MetricsTop + COPA 指标。
    """
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        # weighted regression (scheme 1): weights are set by kfold dataloader / script
        self.pos_weight = float(getattr(args, "pos_weight", getattr(args, "loss_pos_weight", 1.0)))
        self.neg_weight = float(getattr(args, "neg_weight", getattr(args, "loss_neg_weight", 1.0)))
        # control whether to evaluate on dataloader['test'] every epoch in skip_validation mode.
        # For final training (train on all train+valid and only evaluate once on original test),
        # this MUST be False to avoid using test during training/model selection.
        self.test_every_epoch = bool(getattr(args, "test_every_epoch", True))

    def _weighted_l1(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        preds/labels: (B, 1) regression in [-1, 1] ideally
        weight by sign of label (>=0 as positive).
        """
        if self.pos_weight == 1.0 and self.neg_weight == 1.0:
            return torch.mean(torch.abs(preds - labels))
        w = torch.where(labels >= 0, torch.tensor(self.pos_weight, device=labels.device), torch.tensor(self.neg_weight, device=labels.device))
        return torch.mean(w * torch.abs(preds - labels))

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {'train': [], 'valid': [], 'test': []}
        # in skip_validation mode, select best epoch by TEST metric
        key_eval = getattr(self.args, "KeyEval", "Loss")
        min_or_max = 'min' if key_eval in ['Loss'] else 'max'
        best_score = 1e8 if min_or_max == 'min' else -1e8

        max_epochs = int(getattr(self.args, 'max_epochs', 100))
        is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            # nohup/log 文件下不刷 batch 进度，避免日志爆炸
            print(f"\n[CIDerLite] Epoch {epochs}/{max_epochs} (seed={getattr(self.args, 'cur_seed', '-')}) TRAIN...", flush=True)
            with tqdm(dataloader['train'], disable=(not is_tty)) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    optimizer.zero_grad()

                    extras = {}
                    for k in ['ir_feature', 'bio', 'eye', 'eeg', 'eda']:
                        if k in batch_data:
                            extras[k] = batch_data[k].to(self.args.device)

                    lengths = {}
                    if 'audio_lengths' in batch_data:
                        lengths['audio_lengths'] = batch_data['audio_lengths'].to(self.args.device)
                    if 'vision_lengths' in batch_data:
                        lengths['vision_lengths'] = batch_data['vision_lengths'].to(self.args.device)
                    for k in ['ir_feature', 'bio', 'eye', 'eeg', 'eda']:
                        lk = f'{k}_lengths'
                        if lk in batch_data:
                            lengths[lk] = batch_data[lk].to(self.args.device)

                    outputs = model(text, audio, vision, extras=extras, lengths=lengths)['M']
                    # scheme 1: weighted regression loss to avoid collapsing to majority class
                    loss = self._weighted_l1(outputs, labels) if self.args.train_mode != 'classification' else self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            train_results["Loss"] = round(train_loss, 4)
            print(f"[CIDerLite] Epoch {epochs} TRAIN >> {dict_to_str(train_results)}", flush=True)

            # validation
            if getattr(self.args, 'skip_validation', False):
                if self.test_every_epoch:
                    # 每个 epoch 都测试并打印（用于KFold内部选best/观察训练过程）
                    print(f"[CIDerLite] Epoch {epochs} TEST ...", flush=True)
                    test_results = self.do_test(model, dataloader['test'], mode="TEST")
                    brief_keys = [
                        "Loss", "Has0_acc_2", "Non0_acc_2", "Mult_acc_5", "Mult_acc_7", "MAE", "Corr",
                        "COPA_overall_acc",
                    ]
                    brief = {k: test_results.get(k, 0) for k in brief_keys if k in test_results}
                    print(f"[CIDerLite] Epoch {epochs} TEST >> {dict_to_str(brief)}", flush=True)

                    # select best by KeyEval on TEST
                    cur_score = test_results.get(key_eval, None)
                    if cur_score is None:
                        # fallback: Loss
                        cur_score = test_results.get("Loss", None)
                        key_eval_used = "Loss"
                    else:
                        key_eval_used = key_eval

                    if cur_score is not None:
                        is_better = (cur_score <= (best_score - 1e-6)) if min_or_max == 'min' else (cur_score >= (best_score + 1e-6))
                    else:
                        is_better = True  # first time / safety

                    if is_better:
                        best_score = float(cur_score) if cur_score is not None else best_score
                        best_epoch = epochs
                        torch.save(model.cpu().state_dict(), self.args.model_save_path)
                        model.to(self.args.device)
                        print(f"[CIDerLite] Epoch {epochs} >>> SAVE BEST by TEST {key_eval_used}={best_score:.4f}", flush=True)
                    else:
                        print(f"[CIDerLite] Epoch {epochs} keep best_epoch={best_epoch} best_{key_eval_used}={best_score:.4f}", flush=True)

                    if return_epoch_results:
                        epoch_results['train'].append(train_results)
                        epoch_results['test'].append(test_results)
                else:
                    # final-fit mode: do NOT touch test during training. Save the latest weights each epoch.
                    best_epoch = epochs
                    torch.save(model.cpu().state_dict(), self.args.model_save_path)
                    model.to(self.args.device)
                    print(f"[CIDerLite] Epoch {epochs} >>> SAVE (no epoch TEST; final-fit mode)", flush=True)
                    if return_epoch_results:
                        epoch_results['train'].append(train_results)
            else:
                val_results = self.do_test(model, dataloader['valid'], mode="VAL")
                cur_valid = val_results[self.args.KeyEval]
                isBetter = cur_valid <= (best_score - 1e-6) if min_or_max == 'min' else cur_valid >= (best_score + 1e-6)
                if isBetter:
                    best_score, best_epoch = cur_valid, epochs
                    torch.save(model.cpu().state_dict(), self.args.model_save_path)
                    model.to(self.args.device)
                print(f"[CIDerLite] Epoch {epochs} VAL  >> {dict_to_str(val_results)}", flush=True)
                print(f"[CIDerLite] Epoch {epochs} TEST ...", flush=True)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                print(f"[CIDerLite] Epoch {epochs} TEST >> {dict_to_str(test_results)}", flush=True)
                if return_epoch_results:
                    epoch_results['train'].append(train_results)
                    epoch_results['valid'].append(val_results)
                    epoch_results['test'].append(test_results)

            if epochs - best_epoch >= self.args.early_stop or epochs >= getattr(self.args, 'max_epochs', 100):
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        sample_indices = []
        is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader, disable=(not is_tty)) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    extras = {}
                    for k in ['ir_feature', 'bio', 'eye', 'eeg', 'eda']:
                        if k in batch_data:
                            extras[k] = batch_data[k].to(self.args.device)

                    lengths = {}
                    if 'audio_lengths' in batch_data:
                        lengths['audio_lengths'] = batch_data['audio_lengths'].to(self.args.device)
                    if 'vision_lengths' in batch_data:
                        lengths['vision_lengths'] = batch_data['vision_lengths'].to(self.args.device)
                    for k in ['ir_feature', 'bio', 'eye', 'eeg', 'eda']:
                        lk = f'{k}_lengths'
                        if lk in batch_data:
                            lengths[lk] = batch_data[lk].to(self.args.device)

                    outputs = model(text, audio, vision, extras=extras, lengths=lengths)

                    if 'index' in batch_data:
                        batch_indices = batch_data['index'].cpu().numpy()
                        sample_indices.extend(batch_indices)

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        sample_results.extend(preds.squeeze())

                    loss = self.criterion(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        # 对 COPA_1231 来说标签范围是 [-1, 1]（实际为{-1,+1}），这里对评估更稳健：
        pred_clipped = torch.clamp(pred, -1.0, 1.0)
        eval_results = self.metrics(pred_clipped, true)
        eval_results["Loss"] = round(eval_loss, 4)

        # COPA 范式指标
        if self.args.dataset_name.lower() in ['custom', 'train_12_16', 'copa_1231']:
            try:
                copa_metrics = MetricsTop(self.args.train_mode)
                group_type = getattr(self.args, 'copa_group_type', 'i1')
                copa_results = copa_metrics.eval_copa_paradigm_accuracy(
                    pred_clipped, true,
                    sample_indices=np.array(sample_indices) if sample_indices else None,
                    group_type=group_type
                )
                eval_results.update(copa_results)
            except Exception as e:
                logger.warning(f"COPA评估失败: {e}")

        # 自定义 KFold 脚本未配置 logger handler，这里用 print 确保进入 nohup 日志
        print(f"[CIDerLite] {mode} >> {dict_to_str(eval_results)}", flush=True)

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results

