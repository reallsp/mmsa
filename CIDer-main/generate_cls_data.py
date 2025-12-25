"""
生成类别特征和概率文件（用于因果干预）
"""
import pickle
import numpy as np
import os

def generate_cls_data(data_file='./converted_data/copa_pi_aligned_50.pkl',
                      output_dir='./copa_pi_probs',
                      feat_dir='./copa_pi_feats',
                      task='binary'):
    """
    生成类别特征和概率文件
    """
    print("=" * 70)
    print("生成类别特征和概率文件")
    print("=" * 70)
    
    # 1. 加载数据
    print(f"\n1. 加载数据: {data_file}")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    train_data = data['train']
    labels = train_data['classification_labels']
    
    # 2. 计算类别概率
    print("\n2. 计算类别概率...")
    if task == 'binary':
        n_classes = 2
    else:
        n_classes = 7
    
    cls_probs = np.zeros(n_classes, dtype=np.float32)
    for label in labels:
        cls_probs[int(label)] += 1
    cls_probs = cls_probs / len(labels)
    
    print(f"   类别概率: {cls_probs}")
    
    # 3. 计算类别特征（每个类别的平均特征）
    print("\n3. 计算类别特征...")
    
    # 获取特征维度
    text_feat = train_data['text_bert']  # [N, 3, seq_len]
    audio_feat = train_data['audio']  # [N, seq_len, audio_dim]
    vision_feat = train_data['vision']  # [N, seq_len, vision_dim]
    
    # 提取特征（使用平均池化）
    text_features = []
    audio_features = []
    vision_features = []
    
    for i in range(len(labels)):
        # 文本：取平均（简化处理）
        text_vec = text_feat[i].mean(axis=1)  # [3] -> 简化
        text_features.append(text_vec[1])  # 使用input_mask的平均值作为特征
        
        # 音频：取平均
        audio_vec = audio_feat[i].mean(axis=0)  # [audio_dim]
        audio_features.append(audio_vec)
        
        # 视觉：取平均
        vision_vec = vision_feat[i].mean(axis=0)  # [vision_dim]
        vision_features.append(vision_vec)
    
    text_features = np.array(text_features)
    audio_features = np.array(audio_features)
    vision_features = np.array(vision_features)
    
    # 按类别计算平均特征
    cls_feats = {}
    for cls_id in range(n_classes):
        cls_mask = labels == cls_id
        if np.sum(cls_mask) > 0:
            cls_text = text_features[cls_mask].mean(axis=0)
            cls_audio = audio_features[cls_mask].mean(axis=0)
            cls_vision = vision_features[cls_mask].mean(axis=0)
            
            # 组合特征（简化：只使用平均值）
            # 实际应该使用完整的特征，这里简化处理
            cls_feat = np.concatenate([
                np.array([cls_text.mean()]),  # 文本特征的平均值
                cls_audio,  # 音频特征
                cls_vision  # 视觉特征
            ])
            cls_feats[cls_id] = cls_feat
        else:
            # 如果没有该类别，使用零向量
            cls_feats[cls_id] = np.zeros(
                text_features.shape[1] + audio_features.shape[1] + vision_features.shape[1],
                dtype=np.float32
            )
    
    # 4. 保存文件
    print("\n4. 保存文件...")
    
    # 创建目录
    prob_dir = os.path.join(output_dir, 'iid', task)
    feat_dir_path = os.path.join(feat_dir, 'iid', task, 'aligned')
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(feat_dir_path, exist_ok=True)
    
    # 保存概率文件
    prob_file = os.path.join(prob_dir, 'copa_pi_train_binary.npy')
    np.save(prob_file, cls_probs)
    print(f"   概率文件: {prob_file}")
    
    # 保存特征文件（每个类别一个）
    for cls_id in range(n_classes):
        feat_file = os.path.join(feat_dir_path, f'copa_pi_train_binary_aligned_{cls_id}.npy')
        np.save(feat_file, cls_feats[cls_id])
        print(f"   类别 {cls_id} 特征文件: {feat_file}")
    
    print("\n" + "=" * 70)
    print("类别特征和概率文件生成完成！")
    print("=" * 70)

if __name__ == '__main__':
    generate_cls_data(
        data_file='./converted_data/copa_pi_aligned_50.pkl',
        output_dir='./copa_pi_probs',
        feat_dir='./copa_pi_feats',
        task='binary'
    )

