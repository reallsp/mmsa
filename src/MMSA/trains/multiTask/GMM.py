import logging
import os
import pickle as plk

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')

class GMM():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "MTAV"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.center_map = {
            'fusion': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            },
            'text': {
                'pos': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
            },
            'audio': {
                'pos': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
            },
            'vision': {
                'pos': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
            }
        }

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def init_labels(self, indexes, labels):
        for m in self.args.tasks:
            self.label_map[self.name_map[m]][indexes] = labels.float()

    def do_train(self, model, dataloader, return_epoch_results=False):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        saved_labels = {}
        # init labels
        logger.info("Init labels...")
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                indexes = batch_data['index'].view(-1)
                self.init_labels(indexes, labels_m)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths']
                        vision_lengths = batch_data['vision_lengths']
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    # store results
                    for m in self.args.tasks:
                        # Squeeze output if it has shape (batch, 1) to match label shape (batch,)
                        output = outputs[m].cpu()
                        if output.dim() > 1 and output.shape[-1] == 1:
                            output = output.squeeze(-1)
                        y_pred[m].append(output)
                        y_true[m].append(self.label_map[self.name_map[m]][indexes].cpu())
                    # compute loss
                    loss = 0.0
                    for m in self.args.tasks:
                        # Ensure shapes match for loss calculation
                        pred_loss = y_pred[m][-1]
                        true_loss = y_true[m][-1]
                        if pred_loss.dim() > true_loss.dim():
                            pred_loss = pred_loss.squeeze(-1)
                        elif true_loss.dim() > pred_loss.dim():
                            true_loss = true_loss.squeeze(-1)
                        loss += self.args.H * torch.nn.functional.mse_loss(pred_loss, true_loss)
                    loss = loss / len(self.args.tasks)
                    losses.append(loss.item())
                    loss.backward()
                    train_loss += loss.item()

                    if left_epochs == 0:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                optimizer.step()
            # valid
            skip_validation = getattr(self.args, 'skip_validation', False)
            if skip_validation or 'valid' not in dataloader:
                # In KFold mode, use test set as validation set, or skip validation
                if 'test' in dataloader:
                    val_results = self.do_test(model, dataloader['test'], mode="VAL")
                else:
                    # If no test set either, use train metrics as validation
                    train_results = self.metrics(torch.cat(y_pred['M'], dim=0).numpy(), torch.cat(y_true['M'], dim=0).numpy())
                    val_results = train_results
            else:
                val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = (min_or_max == 'min' and cur_valid < best_valid) or (min_or_max == 'max' and cur_valid > best_valid)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.state_dict(), self.args.model_save_path)
            # epoch results
            if return_epoch_results:
                train_results = self.metrics(torch.cat(y_pred['M'], dim=0).numpy(), torch.cat(y_true['M'], dim=0).numpy())
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                logger.info(f"Early stop at epoch {epochs}")
                break
        # test
        model.load_state_dict(torch.load(self.args.model_save_path))
        test_results = self.do_test(model, dataloader['test'], mode="TEST")
        if return_epoch_results:
            epoch_results['test'] = test_results
            return epoch_results
        return test_results

    def do_test(self, model, dataloader, mode="TEST", return_sample_results=False):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = []
        ids = []
        sample_indices = []  # for COPA paradigm accuracy
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)
                    
                    # Collect sample indices for COPA evaluation
                    if 'index' in batch_data:
                        batch_indices = batch_data['index'].cpu().numpy()
                        sample_indices.extend(batch_indices)
                    else:
                        # If no index, use current accumulated count
                        current_start = len(sample_indices)
                        batch_size = indexes.shape[0]
                        sample_indices.extend(range(current_start, current_start + batch_size))

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths']
                        vision_lengths = batch_data['vision_lengths']
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    # store results
                    # Get labels directly from batch_data (test set indexes may not be in label_map)
                    if 'labels' in batch_data and 'M' in batch_data['labels']:
                        labels_m = batch_data['labels']['M'].view(-1).cpu()
                    else:
                        # Fallback to label_map if labels not in batch_data
                        labels_m = self.label_map[self.name_map['M']][indexes].cpu()
                    
                    for m in self.args.tasks:
                        # Squeeze output if it has shape (batch, 1) to match label shape (batch,)
                        output = outputs[m].cpu()
                        if output.dim() > 1 and output.shape[-1] == 1:
                            output = output.squeeze(-1)
                        y_pred[m].append(output)
                        # Use labels_m for all tasks (multi-task models typically use same labels)
                        y_true[m].append(labels_m)
                    # compute loss
                    loss = 0.0
                    for m in self.args.tasks:
                        # Ensure shapes match for loss calculation
                        pred_loss = y_pred[m][-1]
                        true_loss = y_true[m][-1]
                        if pred_loss.dim() > true_loss.dim():
                            pred_loss = pred_loss.squeeze(-1)
                        elif true_loss.dim() > pred_loss.dim():
                            true_loss = true_loss.squeeze(-1)
                        loss += self.args.H * torch.nn.functional.mse_loss(pred_loss, true_loss)
                    loss = loss / len(self.args.tasks)
                    eval_loss.append(loss.item())

        eval_loss = np.mean(eval_loss)
        # Keep tensors for metrics evaluation (metricsTop expects tensors)
        pred, true = torch.cat(y_pred['M'], dim=0), torch.cat(y_true['M'], dim=0)
        eval_results = self.metrics(pred, true)
        eval_results['Loss'] = round(eval_loss, 4)  # Add Loss key for KeyEval
        
        # COPA paradigm metrics (for copa_1231/custom/train_12_16; SCL90 also reuses the same columns)
        if self.args.dataset_name.lower() in ['custom', 'train_12_16', 'copa_1231', 'scl90_1231']:
            try:
                copa_metrics = MetricsTop(self.args.train_mode)
                group_type = getattr(self.args, 'copa_group_type', 'i1')
                # Clip predictions to [-1, 1] for COPA evaluation
                pred_clipped = torch.clamp(pred, -1.0, 1.0)
                copa_results = copa_metrics.eval_copa_paradigm_accuracy(
                    pred_clipped, true,
                    sample_indices=np.array(sample_indices) if sample_indices else None,
                    group_type=group_type
                )
                eval_results.update(copa_results)
            except Exception as e:
                logger.warning(f"COPA评估失败: {e}")
        
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        
        # Convert to numpy for return_sample_results
        if return_sample_results:
            y_pred_numpy = {k: torch.cat(v, 0).numpy() for k, v in y_pred.items()}
            y_true_numpy = {k: torch.cat(v, 0).numpy() for k, v in y_true.items()}
            y_pred_numpy['Loss'] = eval_loss
            return y_pred_numpy, y_true_numpy
        return eval_results
