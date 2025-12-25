"""
数据转换脚本：将 train_12.16_1.pkl 转换为 CIDer 模型期望的格式
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

def convert_text_feature_to_bert_format(text_feature, max_seq_len=50):
    """
    将BERT特征向量转换为伪BERT token格式
    Args:
        text_feature: (768,) BERT特征向量
        max_seq_len: 最大序列长度
    Returns:
        text_bert: [3, max_seq_len] - 伪BERT格式
    """
    # 将特征向量扩展为序列（重复或插值）
    # 方案：将768维特征分成多个token
    feature_dim = text_feature.shape[0]  # 768
    tokens_per_dim = max(1, max_seq_len // 10)  # 每个维度分成多个token
    
    # 创建伪input_ids（使用特征值的整数部分作为token ID）
    # 为了兼容，使用较小的值范围
    input_ids = np.clip((text_feature * 100).astype(np.int32) % 1000, 0, 999)
    input_ids = np.tile(input_ids[:max_seq_len], (max_seq_len // len(input_ids) + 1))[:max_seq_len]
    
    # input_mask: 全部为1（表示有效）
    input_mask = np.ones(max_seq_len, dtype=np.float32)
    
    # segment_ids: 全部为0（单句）
    segment_ids = np.zeros(max_seq_len, dtype=np.int64)
    
    # 组合为 [3, max_seq_len]
    text_bert = np.stack([input_ids, input_mask, segment_ids], axis=0)
    return text_bert.astype(np.float32)

def process_sample(sample, audio_feature_dim=13):
    """
    处理单个样本，转换为CIDer期望的格式
    """
    # 1. 文本特征：转换为伪BERT格式
    text_bert = convert_text_feature_to_bert_format(sample['text'])
    
    # 2. 音频特征：从波形提取MFCC
    audio_features = extract_audio_features(sample['audio_waveform'], n_mfcc=audio_feature_dim)
    
    # 3. 视觉特征：使用video_feature（已经是特征序列）
    vision_features = sample['video_feature']  # (197, 768)
    
    # 4. 标签：转换为期望格式
    label = sample['label']
    # 转换为regression_labels（-1.0到1.0）和classification_labels（0/1）
    regression_label = label * 2.0 - 1.0  # 0.0->-1.0, 1.0->1.0
    classification_label = int(label)  # 0或1
    
    return {
        'text_bert': text_bert,
        'audio': audio_features,
        'vision': vision_features,
        'regression_labels': regression_label,
        'classification_labels': classification_label,
        'raw_text': '',  # 原始文本（如果有的话）
        'id': '',  # 样本ID（可以生成）
    }

def convert_dataset(input_file='train_12.16_1.pkl', 
                   output_dir='./converted_data',
                   train_ratio=0.7,
                   valid_ratio=0.15,
                   test_ratio=0.15):
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
            processed = process_sample(sample)
            processed['id'] = f'sample_{i}'
            processed_samples.append(processed)
        except Exception as e:
            print(f"   样本 {i} 处理失败: {e}")
            continue
    
    print(f"   成功处理: {len(processed_samples)} 个样本")
    
    # 3. 划分数据集
    print("\n3. 划分数据集...")
    labels = [s['classification_labels'] for s in processed_samples]
    
    # 使用分层划分保持标签分布
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        processed_samples, labels, 
        test_size=(1 - train_ratio), 
        stratify=labels,
        random_state=42
    )
    
    valid_ratio_adjusted = valid_ratio / (valid_ratio + test_ratio)
    valid_data, test_data, valid_labels, test_labels = train_test_split(
        temp_data, temp_labels,
        test_size=(1 - valid_ratio_adjusted),
        stratify=temp_labels,
        random_state=42
    )
    
    print(f"   训练集: {len(train_data)} 个样本")
    print(f"   验证集: {len(valid_data)} 个样本")
    print(f"   测试集: {len(test_data)} 个样本")
    
    # 4. 转换为CIDer期望的格式
    print("\n4. 转换为CIDer格式...")
    
    def convert_to_cider_format(samples, mode):
        # 获取所有样本的特征
        text_bert_list = [s['text_bert'] for s in samples]
        audio_list = [s['audio'] for s in samples]
        vision_list = [s['vision'] for s in samples]
        regression_labels = np.array([s['regression_labels'] for s in samples])
        classification_labels = np.array([s['classification_labels'] for s in samples])
        raw_text_list = [s['raw_text'] for s in samples]
        id_list = [s['id'] for s in samples]
        
        # 对齐序列长度（填充或截断）
        # 文本：已经是固定长度
        text_bert = np.array(text_bert_list)  # [N, 3, seq_len]
        
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
        
        return {
            'text_bert': text_bert,
            'audio': audio,
            'vision': vision,
            'raw_text': raw_text_list,
            'id': id_list,
            'regression_labels': regression_labels,
            'classification_labels': classification_labels,
            'audio_lengths': audio_lengths,
            'vision_lengths': vision_lengths,
        }
    
    train_dict = convert_to_cider_format(train_data, 'train')
    valid_dict = convert_to_cider_format(valid_data, 'valid')
    test_dict = convert_to_cider_format(test_data, 'test')
    
    # 5. 保存数据
    print("\n5. 保存数据...")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'copa_pi_aligned_50.pkl')
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
    print(f"   标签分布 - 训练集: {np.bincount(train_dict['classification_labels'])}")
    print(f"   标签分布 - 验证集: {np.bincount(valid_dict['classification_labels'])}")
    print(f"   标签分布 - 测试集: {np.bincount(test_dict['classification_labels'])}")
    
    print("\n" + "=" * 70)
    print("数据转换完成！")
    print("=" * 70)
    
    return output_file, train_dict, valid_dict, test_dict

if __name__ == '__main__':
    # 转换数据
    convert_dataset(
        input_file='../train_12.16_1.pkl',
        output_dir='./converted_data',
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15
    )

