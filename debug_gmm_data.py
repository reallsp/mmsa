#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to check data distribution and understand the "all zeros" issue.
This script doesn't require model weights, just checks the data.
"""

import sys
from pathlib import Path
import torch
import numpy as np

project_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir / "src"))

from MMSA.config import get_config_regression
from MMSA.data_loader_kfold import COPA1231_KFoldDataLoader
from MMSA.trains import ATIO

def main():
    # Setup
    model_name = "gmm"
    dataset_name = "copa_1231"
    fold_idx = 1
    
    print("="*80)
    print("Debugging GMM Test Results - Data Distribution Analysis")
    print("="*80 + "\n")
    
    # Get config
    config = get_config_regression(model_name, dataset_name)
    config['pretrained'] = '/root/autodl-tmp/models/bert-base-uncased'
    config['use_all_features'] = True
    config['batch_size'] = 8
    config['num_workers'] = 0
    config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config['train_mode'] = 'regression'
    
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
    
    # Get trainer for label mapping
    trainer = ATIO().getTrain(config)
    
    # Collect labels from test set
    print("\nAnalyzing test set labels...")
    test_loader = dataloaders['test']
    all_labels = []
    
    for batch_idx, batch_data in enumerate(test_loader):
        indexes = batch_data['index']
        labels = trainer.label_map[trainer.name_map['M']][indexes].cpu()
        all_labels.append(labels.numpy())
        
        # Check all batches
        if batch_idx % 50 == 0:
            print(f"  Processed {batch_idx} batches, {len(np.concatenate(all_labels))} samples so far...")
    
    all_labels = np.concatenate(all_labels)
    
    print(f"\nLabel Statistics:")
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Label range: [{all_labels.min():.4f}, {all_labels.max():.4f}]")
    print(f"  Label mean: {all_labels.mean():.4f}, std: {all_labels.std():.4f}")
    print(f"  Label unique values: {np.unique(all_labels)}")
    print(f"  Labels >= 0: {np.sum(all_labels >= 0)}/{len(all_labels)} ({100*np.sum(all_labels >= 0)/len(all_labels):.2f}%)")
    print(f"  Labels < 0: {np.sum(all_labels < 0)}/{len(all_labels)} ({100*np.sum(all_labels < 0)/len(all_labels):.2f}%)")
    print(f"  Labels == 0: {np.sum(all_labels == 0)}/{len(all_labels)} ({100*np.sum(all_labels == 0)/len(all_labels):.2f}%)")
    
    # Binary classification view
    binary_labels = (all_labels >= 0).astype(int)
    print(f"\nBinary Classification View:")
    print(f"  Class 0 (< 0): {np.sum(binary_labels == 0)} ({100*np.sum(binary_labels == 0)/len(binary_labels):.2f}%)")
    print(f"  Class 1 (>= 0): {np.sum(binary_labels == 1)} ({100*np.sum(binary_labels == 1)/len(binary_labels):.2f}%)")
    
    # Simulate what would happen if all predictions are >= 0
    print(f"\n" + "="*80)
    print("Simulating predictions (all >= 0):")
    print("="*80)
    pred_all_positive = np.ones_like(all_labels) * 0.5  # All predictions = 0.5 (>= 0)
    binary_pred = (pred_all_positive >= 0).astype(int)
    accuracy = np.mean(binary_pred == binary_labels)
    print(f"  Accuracy if all pred >= 0: {accuracy:.4f}")
    
    # Simulate what would happen if all predictions are < 0
    print(f"\nSimulating predictions (all < 0):")
    pred_all_negative = np.ones_like(all_labels) * -0.5  # All predictions = -0.5 (< 0)
    binary_pred = (pred_all_negative >= 0).astype(int)
    accuracy = np.mean(binary_pred == binary_labels)
    print(f"  Accuracy if all pred < 0: {accuracy:.4f}")
    
    # Check what the actual issue might be
    print(f"\n" + "="*80)
    print("Diagnosis:")
    print("="*80)
    if np.sum(all_labels >= 0) == len(all_labels):
        print("  ⚠️  ALL labels are >= 0! This means:")
        print("     - Has0_acc_2 should be comparing pred>=0 vs true>=0")
        print("     - If all predictions are also >= 0, accuracy would be 1.0")
        print("     - If all predictions are < 0, accuracy would be 0.0")
    elif np.sum(all_labels < 0) == len(all_labels):
        print("  ⚠️  ALL labels are < 0! This means:")
        print("     - Has0_acc_2 should be comparing pred>=0 vs true>=0")
        print("     - If all predictions are >= 0, accuracy would be 0.0")
        print("     - If all predictions are < 0, accuracy would be 1.0")
    else:
        print("  ✓ Labels have both positive and negative values")
        print(f"     - Positive: {np.sum(all_labels >= 0)}, Negative: {np.sum(all_labels < 0)}")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("="*80)
    print("  1. Check if model predictions are all the same sign")
    print("  2. Verify the metrics calculation logic")
    print("  3. Check if there's a shape mismatch in predictions")

if __name__ == "__main__":
    main()
