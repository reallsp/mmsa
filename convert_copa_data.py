#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据转换脚本：分开转换训练集和测试集，保留全部特征。
通过显式填充确保每个模态的特征序列长度一致。
"""
import pickle
import numpy as np
import os
import librosa
import warnings
import gc
from pathlib import Path

warnings.filterwarnings('ignore')

def extract_audio_features(audio_waveform, sr=16000, n_mfcc=13, hop_length=512):
    try:
        mfcc = librosa.feature.mfcc(y=audio_waveform, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        return mfcc.T.astype(np.float32)
    except Exception:
        return np.zeros((100, n_mfcc), dtype=np.float32)

def convert_text_to_sequence_format(text_feature, max_seq_len=50):
    if len(text_feature.shape) == 1:
        text_feature = text_feature.reshape(1, -1)
    if text_feature.shape[0] < max_seq_len:
        repeat_times = (max_seq_len // text_feature.shape[0]) + 1
        text_sequence = np.tile(text_feature, (repeat_times, 1))[:max_seq_len, :]
    else:
        text_sequence = text_feature[:max_seq_len, :]
    return text_sequence.astype(np.float32)

def convert_text_to_bert_format(text_feature, max_seq_len=50):
    input_ids = np.clip((text_feature * 100).astype(np.int32) % 1000, 0, 999)
    if len(input_ids) < max_seq_len:
        repeat_times = (max_seq_len // len(input_ids)) + 1
        input_ids = np.tile(input_ids, repeat_times)[:max_seq_len]
    else:
        input_ids = input_ids[:max_seq_len]
    return np.stack([input_ids.astype(np.float32), np.ones(max_seq_len, dtype=np.float32), np.zeros(max_seq_len, dtype=np.float32)], axis=0)

def align_and_stack(feat_list):
    if not feat_list: return None, []
    # 确定最大长度和维度
    max_len = 0
    dim = 0
    for f in feat_list:
        if f.ndim == 1:
            f = f.reshape(-1, 1)
        max_len = max(max_len, f.shape[0])
        dim = f.shape[1]
    
    num_samples = len(feat_list)
    stacked = np.zeros((num_samples, max_len, dim), dtype=np.float32)
    lengths = []
    
    for i, f in enumerate(feat_list):
        if f.ndim == 1:
            f = f.reshape(-1, 1)
        cur_len = min(f.shape[0], max_len)
        stacked[i, :cur_len, :dim] = f[:cur_len, :dim]
        lengths.append(cur_len)
    
    return stacked, lengths

def process_and_save(input_path, output_path, mode):
    print(f"\n>>> 开始处理 {mode} 集: {input_path}")
    with open(input_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    num_samples = len(raw_data)
    print(f"    样本总数: {num_samples}")
    
    # 基础特征
    text_bert = np.zeros((num_samples, 3, 50), dtype=np.float32)
    text = np.zeros((num_samples, 50, 768), dtype=np.float32)
    vision = np.zeros((num_samples, 197, 768), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.float32)
    ids = []
    
    # 变长特征列表
    audio_list = []
    extra_keys = ['ir_feature', 'bio', 'eye', 'eeg', 'eda']
    extra_lists = {k: [] for k in extra_keys if k in raw_data[0]}
    
    for i, s in enumerate(raw_data):
        text_bert[i] = convert_text_to_bert_format(s['text'])
        text[i] = convert_text_to_sequence_format(s['text'])
        vision[i] = s['video_feature'].astype(np.float32)
        labels[i] = s['label'] * 2.0 - 1.0
        ids.append(f"{mode}_{i:05d}")
        
        audio_list.append(extract_audio_features(s['audio_waveform']))
        for k in extra_lists:
            extra_lists[k].append(s[k])
            
        if (i+1) % 2000 == 0:
            print(f"    已处理: {i+1}/{num_samples}")
            gc.collect()
            
    print(f"    正在对齐变长特征...")
    audio_stacked, audio_lengths = align_and_stack(audio_list)
    
    result = {
        'text_bert': text_bert,
        'text': text,
        'audio': audio_stacked,
        'vision': vision,
        'regression_labels': labels,
        'id': ids,
        'raw_text': [''] * num_samples,
        'audio_lengths': audio_lengths,
        'vision_lengths': [197] * num_samples
    }
    
    for k, v_list in extra_lists.items():
        stacked, lengths = align_and_stack(v_list)
        result[k] = stacked
        result[f"{k}_lengths"] = lengths
        
    print(f"    正在保存到: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump({mode: result}, f)
    
    print(f"    {mode} 集处理完成。")
    
    # 显式清理
    del raw_data, text_bert, text, vision, labels, audio_list, extra_lists, audio_stacked, result
    gc.collect()

if __name__ == '__main__':
    data_dir = Path('/root/autodl-tmp/data')
    
    # 分开处理以节省内存
    process_and_save(data_dir / 'copa_train_1231.pkl', data_dir / 'copa_train_1231_converted.pkl', 'train')
    process_and_save(data_dir / 'copa_test_1231.pkl', data_dir / 'copa_test_1231_converted.pkl', 'test')
    
    print("\n✅ 所有转换任务已完成！")
