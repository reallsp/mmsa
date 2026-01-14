import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')

class EF_LSTM():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        best_test = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = model(text, audio, vision)['M']
                    # compute loss
                    loss = self.criterion(outputs, labels)
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
            train_loss = train_loss / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )
            # validation / (KFold) fold-test selection
            if getattr(self.args, 'skip_validation', False):
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                cur_test = test_results[self.args.KeyEval]
                isBetter = cur_test <= (best_test - 1e-6) if min_or_max == 'min' else cur_test >= (best_test + 1e-6)
                if isBetter:
                    best_test, best_epoch = cur_test, epochs
                    torch.save(model.cpu().state_dict(), self.args.model_save_path)
                    model.to(self.args.device)
                if return_epoch_results:
                    train_results["Loss"] = train_loss
                    epoch_results['train'].append(train_results)
                    epoch_results['test'].append(test_results)
            else:
                val_results = self.do_test(model, dataloader['valid'], mode="VAL")
                cur_valid = val_results[self.args.KeyEval]
                isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
                if isBetter:
                    best_valid, best_epoch = cur_valid, epochs
                    torch.save(model.cpu().state_dict(), self.args.model_save_path)
                    model.to(self.args.device)
                if return_epoch_results:
                    train_results["Loss"] = train_loss
                    epoch_results['train'].append(train_results)
                    epoch_results['valid'].append(val_results)
                    test_results = self.do_test(model, dataloader['test'], mode="TEST")
                    epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop or epochs >= getattr(self.args, 'max_epochs', 100):
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        sample_indices = []  # for COPA paradigm accuracy
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
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    outputs = model(text, audio, vision)

                    # collect sample indices for COPA evaluation
                    if 'index' in batch_data:
                        batch_indices = batch_data['index'].cpu().numpy()
                        sample_indices.extend(batch_indices)
                    else:
                        current_start = len(sample_indices)
                        batch_size = labels.shape[0]
                        sample_indices.extend(range(current_start, current_start + batch_size))

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        # TODO: add features
                        # for item in features.keys():
                        #     features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        # test_preds_i = np.argmax(preds, axis=1)
                        sample_results.extend(preds.squeeze())

                    loss = self.criterion(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        # COPA paradigm metrics (for copa_1231/custom/train_12_16; SCL90 also reuses the same columns)
        if self.args.dataset_name.lower() in ['custom', 'train_12_16', 'copa_1231', 'scl90_1231']:
            try:
                copa_metrics = MetricsTop(self.args.train_mode)
                group_type = getattr(self.args, 'copa_group_type', 'i1')
                copa_results = copa_metrics.eval_copa_paradigm_accuracy(
                    pred, true,
                    sample_indices=np.array(sample_indices) if sample_indices else None,
                    group_type=group_type
                )
                eval_results.update(copa_results)
            except Exception as e:
                logger.warning(f"COPA评估失败: {e}")

        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            # for k in features.keys():
            #     features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
