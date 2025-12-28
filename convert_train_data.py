#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据转换脚本：将 train_12.16_1.pkl 转换为 MMSA 框架期望的格式
"""
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
import librosa
import warnings
warnings.filterwarnings('ignore')

def extract_audio_features(audio_waveform, sr=16000, n_mfcc=13, hop_length=512):
    """
    从音频波形提取MFCC特征
    Args:
        audio_waveform: 音频波形数组
        sr: 采样率（默认16000）
        n_mfcc: MFCC特征数量（默认13）
        hop_length: 帧移（默认512）
    Returns:
        mfcc_features: [seq_len, n_mfcc] 特征序列
    """
    try:
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=audio_waveform, 
            sr=sr, 
            n_mfcc=n_mfcc,
            hop_length=hop_length
        )
        # 转置为 [seq_len, n_mfcc]
        mfcc = mfcc.T
        return mfcc.astype(np.float32)
    except Exception as e:
        print(f"音频特征提取失败: {e}")
        # 如果失败，返回零特征
        return np.zeros((100, n_mfcc), dtype=np.float32)

def convert_text_to_sequence_format(text_feature, max_seq_len=50, feature_dim=768):
    """
    将768维BERT特征向量转换为序列格式 [seq_len, dim]
    Args:
        text_feature: (768,) BERT特征向量
        max_seq_len: 目标序列长度
        feature_dim: 目标特征维度 (应为768)
    Returns:
        text_sequence: [max_seq_len, feature_dim] - 序列格式
    """
    # 简单地将768维向量重复或截断为max_seq_len的序列
    if len(text_feature.shape) == 1:
        text_feature = text_feature.reshape(1, -1) # 转换为 (1, 768)
    
    if text_feature.shape[0] < max_seq_len:
        # 重复填充
        repeat_times = (max_seq_len // text_feature.shape[0]) + 1
        text_sequence = np.tile(text_feature, (repeat_times, 1))[:max_seq_len, :]
    else:
        # 截断
        text_sequence = text_feature[:max_seq_len, :]
    
    return text_sequence.astype(np.float32)

def convert_text_to_bert_format(text_feature, max_seq_len=50):
    """
    将768维BERT特征向量转换为伪BERT token格式
    Args:
        text_feature: (768,) BERT特征向量
        max_seq_len: 最大序列长度
    Returns:
        text_bert: [3, max_seq_len] - 伪BERT格式
    """
    # 方案：将768维特征分成多个token
    # 创建伪input_ids（使用特征值的整数部分作为token ID）
    # 为了兼容，使用较小的值范围
    input_ids = np.clip((text_feature * 100).astype(np.int32) % 1000, 0, 999)
    # 如果特征长度小于max_seq_len，重复填充
    if len(input_ids) < max_seq_len:
        repeat_times = (max_seq_len // len(input_ids)) + 1
        input_ids = np.tile(input_ids, repeat_times)[:max_seq_len]
    else:
        input_ids = input_ids[:max_seq_len]
    
    # input_mask: 全部为1（表示有效）
    input_mask = np.ones(max_seq_len, dtype=np.float32)
    
    # segment_ids: 全部为0（单句）
    segment_ids = np.zeros(max_seq_len, dtype=np.int64)
    
    # 组合为 [3, max_seq_len]
    text_bert = np.stack([input_ids, input_mask, segment_ids], axis=0)
    return text_bert.astype(np.float32)

def process_sample(sample, audio_feature_dim=13, text_seq_len=50):
    """
    处理单个样本，转换为MMSA期望的格式
    """
    # 1. 文本特征：转换为伪BERT格式和序列格式
    text_bert = convert_text_to_bert_format(sample['text'], max_seq_len=text_seq_len)
    text_sequence = convert_text_to_sequence_format(sample['text'], max_seq_len=text_seq_len)
    
    # 2. 音频特征：从波形提取MFCC
    audio_features = extract_audio_features(sample['audio_waveform'], n_mfcc=audio_feature_dim)
    
    # 3. 视觉特征：使用video_feature（已经是特征序列）
    vision_features = sample['video_feature']  # (197, 768)
    
    # 4. 标签：转换为期望格式
    label = sample['label']
    # 转换为regression_labels（-1.0到1.0）
    regression_label = label * 2.0 - 1.0  # 0.0->-1.0, 1.0->1.0
    
    result = {
        'text_bert': text_bert,
        'text': text_sequence,  # 序列格式 [seq_len, dim]
        'audio': audio_features,
        'vision': vision_features,
        'regression_labels': regression_label,
        'raw_text': '',  # 原始文本（如果有的话）
        'id': '',  # 样本ID（可以生成）
    }
    
    # 5. 添加其他特征（如果存在）
    if 'ir_feature' in sample:
        result['ir_feature'] = sample['ir_feature'].astype(np.float32)  # (197, 768)
    if 'bio' in sample:
        result['bio'] = sample['bio'].astype(np.float32)  # (4, 3)
    if 'eye' in sample:
        result['eye'] = sample['eye'].astype(np.float32)  # (48, 2)
    if 'eeg' in sample:
        result['eeg'] = sample['eeg'].astype(np.float32)  # (14, 8)
    if 'eda' in sample:
        result['eda'] = sample['eda'].astype(np.float32)  # (675, 7)
    
    return result

def convert_dataset(input_file='train_12.16_1.pkl', 
                   output_file='train_12.16_1_converted.pkl',
                   train_ratio=0.7,
                   valid_ratio=0.15,
                   test_ratio=0.15,
                   audio_feature_dim=13,
                   text_seq_len=50):
    """
    转换整个数据集
    """
    print("=" * 70)
    print("开始数据转换")
    print("=" * 70)
    
    # 1. 加载原始数据
    print(f"\n1. 加载原始数据: {input_file}")
    with open(input_file, 'rb') as f:
        raw_data = pickle.load(f)
    print(f"   总样本数: {len(raw_data)}")
    
    # 2. 处理每个样本
    print("\n2. 处理样本...")
    processed_samples = []
    for i, sample in enumerate(raw_data):
        if (i + 1) % 50 == 0:
            print(f"   处理进度: {i+1}/{len(raw_data)}")
        try:
            processed = process_sample(sample, audio_feature_dim, text_seq_len)
            processed['id'] = f'sample_{i:04d}'
            processed_samples.append(processed)
        except Exception as e:
            print(f"   样本 {i} 处理失败: {e}")
            continue
    
    print(f"   成功处理: {len(processed_samples)} 个样本")
    
    # 3. 划分数据集
    print("\n3. 划分数据集...")
    labels = [s['regression_labels'] for s in processed_samples]
    
    # 使用分层划分保持标签分布（将回归标签转换为分类标签用于分层）
    classification_labels = [int((l + 1) / 2) for l in labels]  # -1->0, 1->1
    
    # 使用新的随机种子重新划分数据集（从42改为123，确保划分不同）
    split_seed = 123  # 修改此值可以改变数据集划分
    print(f"   使用随机种子: {split_seed}")
    
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        processed_samples, classification_labels, 
        test_size=(1 - train_ratio), 
        stratify=classification_labels,
        random_state=split_seed
    )
    
    valid_ratio_adjusted = valid_ratio / (valid_ratio + test_ratio)
    valid_data, test_data, valid_labels, test_labels = train_test_split(
        temp_data, temp_labels,
        test_size=(1 - valid_ratio_adjusted),
        stratify=temp_labels,
        random_state=split_seed + 100  # 使用不同的种子确保随机性
    )
    
    print(f"   训练集: {len(train_data)} 个样本")
    print(f"   验证集: {len(valid_data)} 个样本")
    print(f"   测试集: {len(test_data)} 个样本")
    
    # 4. 转换为MMSA期望的格式
    print("\n4. 转换为MMSA格式...")
    
    def convert_to_mmsa_format(samples, mode):
        # 获取所有样本的特征
        text_bert_list = [s['text_bert'] for s in samples]
        text_list = [s['text'] for s in samples]  # 获取text序列
        audio_list = [s['audio'] for s in samples]
        vision_list = [s['vision'] for s in samples]
        regression_labels = np.array([s['regression_labels'] for s in samples])
        raw_text_list = [s['raw_text'] for s in samples]
        id_list = [s['id'] for s in samples]
        
        # 检查是否有额外特征
        has_ir = 'ir_feature' in samples[0] if len(samples) > 0 else False
        has_bio = 'bio' in samples[0] if len(samples) > 0 else False
        has_eye = 'eye' in samples[0] if len(samples) > 0 else False
        has_eeg = 'eeg' in samples[0] if len(samples) > 0 else False
        has_eda = 'eda' in samples[0] if len(samples) > 0 else False
        
        # 对齐序列长度（填充或截断）
        # 文本BERT格式：已经是固定长度 [3, seq_len]
        text_bert = np.array(text_bert_list)  # [N, 3, seq_len]
        # 文本序列格式：已经是固定长度 [seq_len, dim]
        text = np.array(text_list)  # [N, seq_len, dim]
        
        # 音频：需要对齐
        max_audio_len = max([a.shape[0] for a in audio_list])
        audio_dim = audio_list[0].shape[1]
        audio_aligned = []
        audio_lengths = []
        for a in audio_list:
            audio_len = a.shape[0]
            audio_lengths.append(audio_len)
            if audio_len < max_audio_len:
                # 填充
                padding = np.zeros((max_audio_len - audio_len, audio_dim), dtype=np.float32)
                a_aligned = np.vstack([a, padding])
            else:
                # 截断
                a_aligned = a[:max_audio_len]
            audio_aligned.append(a_aligned)
        audio = np.array(audio_aligned)  # [N, max_seq_len, audio_dim]
        
        # 视觉：已经是固定长度 (197, 768)
        vision = np.array(vision_list)  # [N, 197, 768]
        vision_lengths = [197] * len(samples)
        
        result = {
            'text_bert': text_bert,
            'text': text,  # 序列格式 [N, seq_len, dim]
            'audio': audio,
            'vision': vision,
            'raw_text': raw_text_list,
            'id': id_list,
            'regression_labels': regression_labels,
            'audio_lengths': audio_lengths,
            'vision_lengths': vision_lengths,
        }
        
        # 添加额外特征（如果存在）
        if has_ir:
            ir_list = [s['ir_feature'] for s in samples]
            result['ir_feature'] = np.array(ir_list)  # [N, 197, 768]
            result['ir_lengths'] = [197] * len(samples)
        
        if has_bio:
            bio_list = [s['bio'] for s in samples]
            # bio特征需要对齐（因为长度可能不同）
            max_bio_len = max([b.shape[0] for b in bio_list])
            bio_dim = bio_list[0].shape[1]
            bio_aligned = []
            bio_lengths = []
            for b in bio_list:
                bio_len = b.shape[0]
                bio_lengths.append(bio_len)
                if bio_len < max_bio_len:
                    padding = np.zeros((max_bio_len - bio_len, bio_dim), dtype=np.float32)
                    b_aligned = np.vstack([b, padding])
                else:
                    b_aligned = b[:max_bio_len]
                bio_aligned.append(b_aligned)
            result['bio'] = np.array(bio_aligned)  # [N, max_seq_len, 3]
            result['bio_lengths'] = bio_lengths
        
        if has_eye:
            eye_list = [s['eye'] for s in samples]
            # eye特征需要对齐
            max_eye_len = max([e.shape[0] for e in eye_list])
            eye_dim = eye_list[0].shape[1]
            eye_aligned = []
            eye_lengths = []
            for e in eye_list:
                eye_len = e.shape[0]
                eye_lengths.append(eye_len)
                if eye_len < max_eye_len:
                    padding = np.zeros((max_eye_len - eye_len, eye_dim), dtype=np.float32)
                    e_aligned = np.vstack([e, padding])
                else:
                    e_aligned = e[:max_eye_len]
                eye_aligned.append(e_aligned)
            result['eye'] = np.array(eye_aligned)  # [N, max_seq_len, 2]
            result['eye_lengths'] = eye_lengths
        
        if has_eeg:
            eeg_list = [s['eeg'] for s in samples]
            # eeg特征需要对齐
            max_eeg_len = max([e.shape[0] for e in eeg_list])
            eeg_dim = eeg_list[0].shape[1]
            eeg_aligned = []
            eeg_lengths = []
            for e in eeg_list:
                eeg_len = e.shape[0]
                eeg_lengths.append(eeg_len)
                if eeg_len < max_eeg_len:
                    padding = np.zeros((max_eeg_len - eeg_len, eeg_dim), dtype=np.float32)
                    e_aligned = np.vstack([e, padding])
                else:
                    e_aligned = e[:max_eeg_len]
                eeg_aligned.append(e_aligned)
            result['eeg'] = np.array(eeg_aligned)  # [N, max_seq_len, 8]
            result['eeg_lengths'] = eeg_lengths
        
        if has_eda:
            eda_list = [s['eda'] for s in samples]
            # EDA特征需要对齐（因为长度可能不同）
            max_eda_len = max([e.shape[0] for e in eda_list])
            eda_dim = eda_list[0].shape[1]
            eda_aligned = []
            eda_lengths = []
            for e in eda_list:
                eda_len = e.shape[0]
                eda_lengths.append(eda_len)
                if eda_len < max_eda_len:
                    padding = np.zeros((max_eda_len - eda_len, eda_dim), dtype=np.float32)
                    e_aligned = np.vstack([e, padding])
                else:
                    e_aligned = e[:max_eda_len]
                eda_aligned.append(e_aligned)
            result['eda'] = np.array(eda_aligned)  # [N, max_seq_len, 7]
            result['eda_lengths'] = eda_lengths
        
        return result
    
    train_dict = convert_to_mmsa_format(train_data, 'train')
    valid_dict = convert_to_mmsa_format(valid_data, 'valid')
    test_dict = convert_to_mmsa_format(test_data, 'test')
    
    # 5. 保存数据
    print("\n5. 保存数据...")
    output_data = {
        'train': train_dict,
        'valid': valid_dict,
        'test': test_dict,
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"   保存到: {output_file}")
    
    # 6. 打印统计信息
    print("\n6. 数据统计:")
    print(f"   文本特征形状: {train_dict['text_bert'].shape}")
    print(f"   音频特征形状: {train_dict['audio'].shape}")
    print(f"   视觉特征形状: {train_dict['vision'].shape}")
    print(f"   标签范围 - 训练集: [{train_dict['regression_labels'].min():.2f}, {train_dict['regression_labels'].max():.2f}]")
    print(f"   标签范围 - 验证集: [{valid_dict['regression_labels'].min():.2f}, {valid_dict['regression_labels'].max():.2f}]")
    print(f"   标签范围 - 测试集: [{test_dict['regression_labels'].min():.2f}, {test_dict['regression_labels'].max():.2f}]")
    print(f"   音频序列长度范围: [{min(train_dict['audio_lengths'])}, {max(train_dict['audio_lengths'])}]")
    
    # 打印额外特征信息
    extra_features = []
    if 'ir_feature' in train_dict:
        extra_features.append(f"红外特征(ir_feature): {train_dict['ir_feature'].shape}")
    if 'bio' in train_dict:
        extra_features.append(f"生物特征(bio): {train_dict['bio'].shape}")
    if 'eye' in train_dict:
        extra_features.append(f"眼动特征(eye): {train_dict['eye'].shape}")
    if 'eeg' in train_dict:
        extra_features.append(f"脑电特征(eeg): {train_dict['eeg'].shape}")
    if 'eda' in train_dict:
        extra_features.append(f"皮肤电特征(eda): {train_dict['eda'].shape}")
        print(f"   EDA序列长度范围: [{min(train_dict['eda_lengths'])}, {max(train_dict['eda_lengths'])}]")
    
    if extra_features:
        print("\n   额外特征:")
        for feat in extra_features:
            print(f"     {feat}")
    
    print("\n" + "=" * 70)
    print("数据转换完成！")
    print("=" * 70)
    
    return output_file, train_dict, valid_dict, test_dict

if __name__ == '__main__':
    # 转换数据
    convert_dataset(
        input_file='train_12.16_1.pkl',
        output_file='train_12.16_1_converted.pkl',
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15,
        audio_feature_dim=13,
        text_seq_len=50
    )

