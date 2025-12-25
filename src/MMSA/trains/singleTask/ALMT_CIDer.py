"""
ALMT_CIDer训练器

支持模块化设计的多模态情感分析训练器，可灵活启用/关闭增强模块：
- CIDer自蒸馏
- 智能特征对齐  （弃用-没有效果，因为原始的模块已经提供相关对齐功能）
- 模态贡献平衡
- 双向交叉注意力
"""

import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from ...utils import MetricsTop, dict_to_str
from transformers import BertTokenizer
from .ALMT import get_scheduler
import numpy as np

logger = logging.getLogger('MMSA')

class ALMT_CIDer():
    
    def __init__(self, args):
        self.args = args
        
        # 初始化配置
        self._init_config()
        
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained)
        self.criterion = nn.MSELoss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        
        # 模块启用状态
        self.enable_cider_distill = getattr(args, 'enable_cider_distillation', True)
        self.enable_alignment = getattr(args, 'enable_intelligent_alignment', True)  
        self.enable_contribution_balance = getattr(args, 'enable_modality_balance', True)
        self.enable_bidirectional_attn = getattr(args, 'enable_bidirectional_attention', True)
        
        # 打印模块状态
        logger.info(f"ALMT-CIDer训练器初始化完成")
        logger.info(f"   启用模块:")
        logger.info(f"      CIDer自蒸馏: {'是' if self.enable_cider_distill else '否'}")
        logger.info(f"      智能对齐: {'是' if self.enable_alignment else '否'}")
        logger.info(f"      模态平衡: {'是' if self.enable_contribution_balance else '否'}")
        logger.info(f"      双向交叉注意力: {'是' if self.enable_bidirectional_attn else '否'}")
        logger.info(f"   损失权重: {self.loss_weights}")
    
    def _init_config(self):
        """初始化配置"""
        # 默认损失权重
        self.loss_weights = {
            'task_loss': 1.0,
            'distillation_loss': 0.5,
            'alignment_loss': 0.3,
            'contribution_loss': 0.2,
            'cross_attention_loss': 0.1
        }
        
        # 从YAML配置中加载训练配置
        if hasattr(self.args, '_yaml_config'):
            config = self.args._yaml_config
            training_config = config.get('training_config', {})
            
            # 更新损失权重
            loss_weights = training_config.get('loss_weights', {})
            self.loss_weights.update(loss_weights)
    
    def do_train(self, model, dataloader, return_epoch_results=False):
        """训练方法"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        scheduler_warmup = get_scheduler(optimizer, self.args.max_epochs)

        # 初始化结果
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {'train': [], 'valid': [], 'test': []}
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True: 
            epochs += 1
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            
            with tqdm(dataloader['train']) as td:
                for index, batch_data in enumerate(td):
                    loss = 0.0
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M']
                    labels = labels.to(self.args.device).view(-1, 1)
                    
                    optimizer.zero_grad()
                    
                    # 调用模型
                    outputs = model(text, audio, vision)
                    
                    # 处理输出
                    if isinstance(outputs, dict):
                        logits = outputs['prediction']
                    else:
                        logits = outputs
                    
                    loss += self.criterion(logits, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    logits = logits.detach().cpu()
                    labels = labels.detach().cpu()
                    train_loss += loss.item()
                    y_pred.append(logits)
                    y_true.append(labels)
            
            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d)>> loss: %.4f " % (self.args.model_name, epochs - best_epoch, epochs, train_loss))
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info('%s: >> ' %('Multimodal') + dict_to_str(train_results))
            
            # 验证
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            scheduler_warmup.step()
            
            # 保存最佳模型
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            
            # 早停检查
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def _init_loss_details(self):
        """初始化损失统计字典"""
        return {
            'task_loss': 0.0,
            'distillation_loss': 0.0,
            'alignment_loss': 0.0,
            'contribution_loss': 0.0,
            'cross_attention_loss': 0.0,
            'total_loss': 0.0
        }

    def _compute_weighted_losses(self, outputs, task_loss):
        """计算加权损失"""
        weighted_losses = {'task_loss': task_loss * self.loss_weights['task_loss']}
        
        # 蒸馏损失
        if self.enable_cider_distill and 'distillation_loss' in outputs:
            weighted_losses['distillation_loss'] = (
                outputs['distillation_loss'] * self.loss_weights['distillation_loss']
            )
        else:
            weighted_losses['distillation_loss'] = torch.tensor(0.0).to(task_loss.device)
        
        # 对齐损失
        if (self.enable_alignment or self.enable_bidirectional_attn) and 'alignment_loss' in outputs:
            weighted_losses['alignment_loss'] = (
                outputs['alignment_loss'] * self.loss_weights['alignment_loss']
            )
        else:
            weighted_losses['alignment_loss'] = torch.tensor(0.0).to(task_loss.device)
        
        # 贡献损失
        if self.enable_contribution_balance and 'contribution_loss' in outputs:
            weighted_losses['contribution_loss'] = (
                outputs['contribution_loss'] * self.loss_weights['contribution_loss']
            )
        else:
            weighted_losses['contribution_loss'] = torch.tensor(0.0).to(task_loss.device)
        
        # 交叉注意力损失
        if self.enable_bidirectional_attn and 'cross_attention_loss' in outputs:
            weighted_losses['cross_attention_loss'] = (
                outputs['cross_attention_loss'] * self.loss_weights['cross_attention_loss']
            )
        else:
            weighted_losses['cross_attention_loss'] = torch.tensor(0.0).to(task_loss.device)
        
        return weighted_losses

    def _train_epoch(self, model, dataloader, optimizer, epoch):
        """训练单个epoch"""
        model.train()
        
        # 检查当前模式
        is_baseline_mode = not any([
            self.enable_cider_distill, self.enable_alignment,
            self.enable_contribution_balance, self.enable_bidirectional_attn
        ])
        
        y_pred, y_true = [], []
        loss_details = self._init_loss_details()
        
        # 构建描述信息
        if is_baseline_mode:
            desc = f"Baseline模式 - Epoch {epoch}"
        else:
            active_modules = []
            if self.enable_cider_distill:
                active_modules.append("自蒸馏")
            if self.enable_alignment:
                active_modules.append("对齐")
            if self.enable_contribution_balance:
                active_modules.append("平衡")
            if self.enable_bidirectional_attn:
                active_modules.append("双向注意力")
            desc = f"增强模式({'+'.join(active_modules)}) - Epoch {epoch}"
        
        with tqdm(dataloader, desc=desc) as td:
            for batch_data in td:
                # 数据准备
                vision = batch_data['vision'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                text = batch_data['text'].to(self.args.device)
                labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)
                
                optimizer.zero_grad()
                
                if is_baseline_mode:
                    # Baseline模式：完全模拟标准ALMT
                    outputs = model(text, audio, vision)
                    logits = outputs
                    task_loss = self.criterion(logits, labels)
                    total_loss = task_loss
                    
                    # 简化的损失统计
                    loss_details['task_loss'] += task_loss.item()
                    loss_details['total_loss'] += total_loss.item()
                    
                else:
                    # 增强模式：使用所有启用的模块
                    outputs = model(text, audio, vision, labels=labels, training=True)
                    
                    if isinstance(outputs, dict):
                        prediction = outputs['prediction']
                        task_loss = self.criterion(prediction, labels)
                        
                        # 计算加权损失
                        weighted_losses = self._compute_weighted_losses(outputs, task_loss)
                        total_loss = sum(weighted_losses.values())
                        
                        # 统计损失
                        for key, value in weighted_losses.items():
                            loss_details[key] += value.item()
                        loss_details['total_loss'] += total_loss.item()
                        
                    else:
                        # 回退到简单模式
                        prediction = outputs
                        task_loss = self.criterion(prediction, labels)
                        total_loss = task_loss
                        
                        loss_details['task_loss'] += task_loss.item()
                        loss_details['total_loss'] += total_loss.item()
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                max_grad_norm = getattr(self.args, 'max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # 优化器步进
                optimizer.step()
                
                # 收集预测结果
                prediction = prediction.detach().cpu()
                labels = labels.detach().cpu()
                y_pred.append(prediction)
                y_true.append(labels)
        
        # 计算训练指标
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        train_results = self.metrics(pred, true)
        
        # 添加损失信息
        for key, value in loss_details.items():
            train_results[key] = value / len(dataloader)
        
        return train_results
    
    def do_test(self, model, dataloader, mode="VAL"):
        """测试方法"""
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)
                    
                    # 调用模型
                    outputs = model(text, audio, vision)
                    
                    # 处理输出
                    if isinstance(outputs, dict):
                        logits = outputs['prediction']
                    else:
                        logits = outputs
                    
                    loss = self.criterion(logits, labels)
                    
                    eval_loss += loss.item()
                    logits = logits.detach().cpu()
                    labels = labels.detach().cpu()
                    y_pred.append(logits)
                    y_true.append(labels)
        
        eval_loss = round(eval_loss / len(dataloader), 4)
        logger.info(mode+"-(%s)" % self.args.model_name + " >> loss: %.4f " % eval_loss)
        
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        results = self.metrics(pred, true)
        logger.info('%s: >> ' %('Multimodal') + dict_to_str(results))
        
        eval_results = results
        eval_results['Loss'] = eval_loss
        return eval_results
    
    def _format_results(self, results):
        """格式化结果显示"""
        main_metrics = []
        for key in ['Mult_acc_7', 'Mult_acc_2', 'F1_score', 'MAE', 'Corr']:
            if key in results:
                main_metrics.append(f"{key}: {results[key]:.4f}")
        
        loss_info = []
        if 'total_loss' in results:
            loss_info.append(f"Total: {results['total_loss']:.4f}")
        if 'task_loss' in results:
            loss_info.append(f"Task: {results['task_loss']:.4f}")
        
        # 显示启用模块的损失
        if self.enable_cider_distill and 'distillation_loss' in results and results['distillation_loss'] > 0.0001:
            loss_info.append(f"Distill: {results['distillation_loss']:.4f}")
        if self.enable_alignment and 'alignment_loss' in results and results['alignment_loss'] > 0.0001:
            loss_info.append(f"Align: {results['alignment_loss']:.4f}")
        if self.enable_contribution_balance and 'contribution_loss' in results and results['contribution_loss'] > 0.0001:
            loss_info.append(f"Contrib: {results['contribution_loss']:.4f}")
        if self.enable_bidirectional_attn and 'cross_attention_loss' in results and results['cross_attention_loss'] > 0.0001:
            loss_info.append(f"CrossAttn: {results['cross_attention_loss']:.4f}")
        
        result_str = " | ".join(main_metrics)
        if loss_info:
            result_str += f" || Loss: {', '.join(loss_info)}"
            
        return result_str
