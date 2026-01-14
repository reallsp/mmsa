#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to test GMM model with saved weights and diagnose the "all zeros" issue.
"""

import sys
from pathlib import Path
import torch
import numpy as np

project_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir / "src"))

from MMSA.config import get_config_regression
from MMSA.data_loader_kfold import COPA1231_KFoldDataLoader
from MMSA.models import AMIO
from MMSA.trains import ATIO
from MMSA.utils.experiment import configure_mmsa_logger

def main():
    # Setup
    model_name = "gmm"
    dataset_name = "copa_1231"
    fold_idx = 1
    model_path = f"saved_models/fold{fold_idx}/gmm-copa_1231.pth"
    
    print(f"Loading model from: {model_path}")
    
    # Get config
    config = get_config_regression(model_name, dataset_name)
    config['pretrained'] = '/root/autodl-tmp/models/bert-base-uncased'
    config['use_all_features'] = True
    config['batch_size'] = 8
    config['num_workers'] = 0
    config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("Loading data...")
    dataloaders, _ = COPA1231_KFoldDataLoader(
        config,
        num_workers=0,
        fold_idx=fold_idx,
        n_splits=5,
        split_seed=1111,
        include_test_part=False,
    )
    
    # Load model
    print("Loading model...")
    model = AMIO(config).to(config['device'])
    model.load_state_dict(torch.load(model_path, map_location=config['device']))
    model.eval()
    
    # Get trainer
    trainer = ATIO().getTrain(config)
    
    # Run test with detailed debugging
    print("\n" + "="*80)
    print("Running test with detailed debugging...")
    print("="*80 + "\n")
    
    test_loader = dataloaders['test']
    y_pred = {'M': []}
    y_true = {'M': []}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if batch_idx >= 5:  # Only check first 5 batches for debugging
                break
                
            text = batch_data['text'].to(config['device'])
            audio = batch_data['audio'].to(config['device'])
            vision = batch_data['vision'].to(config['device'])
            labels = batch_data['labels']['M'].to(config['device'])
            
            # Get indexes for label mapping
            indexes = batch_data['index']
            
            # Forward
            if isinstance(audio, tuple):
                audio, audio_lengths = audio
            else:
                audio_lengths = None
            if isinstance(vision, tuple):
                vision, vision_lengths = vision
            else:
                vision_lengths = None
            
            outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
            
            # Store predictions and labels
            pred = outputs['M'].cpu()
            true = trainer.label_map[trainer.name_map['M']][indexes].cpu()
            
            # Debug: print shapes and values
            print(f"\nBatch {batch_idx}:")
            print(f"  pred shape: {pred.shape}, true shape: {true.shape}")
            print(f"  pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}], mean: {pred.mean().item():.4f}")
            print(f"  true range: [{true.min().item():.4f}, {true.max().item():.4f}], mean: {true.mean().item():.4f}")
            print(f"  pred values (first 8): {pred.squeeze()[:8] if pred.dim() > 1 else pred[:8]}")
            print(f"  true values (first 8): {true[:8]}")
            print(f"  pred >= 0: {torch.sum(pred >= 0).item()}/{pred.numel()}")
            print(f"  true >= 0: {torch.sum(true >= 0).item()}/{true.numel()}")
            
            # Squeeze if needed
            if pred.dim() > 1 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            if true.dim() > 1 and true.shape[-1] == 1:
                true = true.squeeze(-1)
            
            y_pred['M'].append(pred)
            y_true['M'].append(true)
    
    # Concatenate and evaluate
    print("\n" + "="*80)
    print("Overall Statistics:")
    print("="*80 + "\n")
    
    pred_all = torch.cat(y_pred['M'], dim=0)
    true_all = torch.cat(y_true['M'], dim=0)
    
    pred_np = pred_all.numpy().flatten()
    true_np = true_all.numpy().flatten()
    
    print(f"Total samples: {len(pred_np)}")
    print(f"Pred range: [{pred_np.min():.4f}, {pred_np.max():.4f}], mean: {pred_np.mean():.4f}, std: {pred_np.std():.4f}")
    print(f"True range: [{true_np.min():.4f}, {true_np.max():.4f}], mean: {true_np.mean():.4f}, std: {true_np.std():.4f}")
    print(f"Pred unique values (first 20): {np.unique(pred_np)[:20]}")
    print(f"True unique values: {np.unique(true_np)}")
    print(f"\nPred >= 0: {np.sum(pred_np >= 0)}/{len(pred_np)} ({100*np.sum(pred_np >= 0)/len(pred_np):.2f}%)")
    print(f"True >= 0: {np.sum(true_np >= 0)}/{len(true_np)} ({100*np.sum(true_np >= 0)/len(true_np):.2f}%)")
    print(f"Pred < 0: {np.sum(pred_np < 0)}/{len(pred_np)} ({100*np.sum(pred_np < 0)/len(pred_np):.2f}%)")
    print(f"True < 0: {np.sum(true_np < 0)}/{len(true_np)} ({100*np.sum(true_np < 0)/len(true_np):.2f}%)")
    
    # Binary classification stats
    binary_pred = (pred_np >= 0).astype(int)
    binary_true = (true_np >= 0).astype(int)
    print(f"\nBinary classification:")
    print(f"  Pred class 0: {np.sum(binary_pred == 0)}, class 1: {np.sum(binary_pred == 1)}")
    print(f"  True class 0: {np.sum(binary_true == 0)}, class 1: {np.sum(binary_true == 1)}")
    print(f"  Accuracy: {np.mean(binary_pred == binary_true):.4f}")
    
    # Run metrics
    print("\n" + "="*80)
    print("Running metrics evaluation...")
    print("="*80 + "\n")
    
    eval_results = trainer.metrics(pred_all, true_all)
    for key, value in eval_results.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("Debugging complete!")
    print("="*80)

if __name__ == "__main__":
    main()
