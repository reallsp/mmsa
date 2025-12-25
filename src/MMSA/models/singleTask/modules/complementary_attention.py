"""
Complementary Attention Mechanism
互补注意力机制 - ALMT-CAF的核心创新

基于CrossFuse论文思想：关注模态间的互补性而非相似性
论文: CrossFuse: A Novel Cross Attention Mechanism based Infrared and Visible Image Fusion Approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys
from pathlib import Path

# 使用绝对路径导入
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))


class ComplementaryAttention(nn.Module):
    """
    互补注意力层
    核心思想: softmax(-Q·K^T) 关注互补特征而非相似特征
    """
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, temperature=1.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.temperature = temperature
        self.attend = nn.Softmax(dim=-1)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            output: [batch_size, seq_len, dim]
            attention_weights: [batch_size, heads, seq_len, seq_len]
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 核心创新: 使用负的注意力分数关注互补性
        complementary_dots = -dots / self.temperature
        attn = self.attend(complementary_dots)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out), attn


class ModalityQualityAssessor(nn.Module):
    """
    模态质量评估器
    评估各模态的质量并动态调整注意力权重
    """
    
    def __init__(self, dim):
        super().__init__()
        self.quality_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            quality_score: [batch_size, 1, 1]
        """
        # 使用全局平均池化获取模态表示
        pooled = x.mean(dim=1)  # [batch_size, dim]
        quality = self.quality_net(pooled)  # [batch_size, 1]
        return quality.unsqueeze(-1)  # [batch_size, 1, 1]


class AdaptiveComplementaryFusion(nn.Module):
    """
    自适应互补融合模块
    结合传统注意力和互补注意力，并进行质量加权
    """
    
    def __init__(self, dim, heads=8, dropout=0.0, alpha=0.6):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))  # 可学习的平衡参数
        
        # 传统注意力 (相似性)
        self.similarity_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        
        # 互补注意力 (差异性)
        self.complementary_attention = ComplementaryAttention(
            dim=dim, heads=heads, dropout=dropout
        )
        
        # 模态质量评估
        self.quality_assessor = ModalityQualityAssessor(dim)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
    def forward(self, query, key, value):
        """
        Args:
            query, key, value: [batch_size, seq_len, dim]
        Returns:
            fused_output: [batch_size, seq_len, dim]
            attention_info: dict containing attention weights and quality scores
        """
        # 1. 传统相似性注意力
        sim_out, sim_attn = self.similarity_attention(query, key, value)
        
        # 2. 互补性注意力 (针对query)
        comp_out, comp_attn = self.complementary_attention(query)
        
        # 3. 模态质量评估
        quality_score = self.quality_assessor(query)
        
        # 4. 自适应权重融合
        alpha = torch.sigmoid(self.alpha)  # 确保在[0,1]范围内
        adaptive_out = alpha * sim_out + (1 - alpha) * comp_out
        
        # 5. 质量加权
        quality_weighted = adaptive_out * quality_score
        
        # 6. 残差连接和最终融合
        concat_features = torch.cat([query, quality_weighted], dim=-1)
        fused_output = self.fusion_layer(concat_features) + query
        
        # 收集注意力信息用于分析
        attention_info = {
            'similarity_attention': sim_attn,
            'complementary_attention': comp_attn,
            'quality_score': quality_score,
            'alpha': alpha.item()
        }
        
        return fused_output, attention_info


class CrossModalComplementaryFusion(nn.Module):
    """
    跨模态互补融合模块
    处理文本、音频、视觉三个模态间的互补关系
    """
    
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        
        # 三个模态的互补融合层
        self.text_fusion = AdaptiveComplementaryFusion(dim, heads, dropout)
        self.audio_fusion = AdaptiveComplementaryFusion(dim, heads, dropout)
        self.vision_fusion = AdaptiveComplementaryFusion(dim, heads, dropout)
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
    def forward(self, text_feat, audio_feat, vision_feat):
        """
        Args:
            text_feat, audio_feat, vision_feat: [batch_size, seq_len, dim]
        Returns:
            fused_features: [batch_size, seq_len, dim]
            attention_analysis: dict containing detailed attention information
        """
        # 1. 各模态内部的互补注意力
        text_enhanced, text_info = self.text_fusion(text_feat, text_feat, text_feat)
        audio_enhanced, audio_info = self.audio_fusion(audio_feat, audio_feat, audio_feat)
        vision_enhanced, vision_info = self.vision_fusion(vision_feat, vision_feat, vision_feat)
        
        # 2. 跨模态互补融合
        # 文本引导音频和视觉
        audio_text_guided, audio_text_info = self.audio_fusion(audio_feat, text_feat, text_feat)
        vision_text_guided, vision_text_info = self.vision_fusion(vision_feat, text_feat, text_feat)
        
        # 音频引导文本和视觉  
        text_audio_guided, text_audio_info = self.text_fusion(text_feat, audio_feat, audio_feat)
        vision_audio_guided, vision_audio_info = self.vision_fusion(vision_feat, audio_feat, audio_feat)
        
        # 3. 特征聚合
        text_final = (text_enhanced + text_audio_guided) / 2
        audio_final = (audio_enhanced + audio_text_guided) / 2  
        vision_final = (vision_enhanced + vision_text_guided + vision_audio_guided) / 3
        
        # 4. 最终融合
        concat_features = torch.cat([text_final, audio_final, vision_final], dim=-1)
        fused_features = self.final_fusion(concat_features)
        
        # 5. 收集分析信息
        attention_analysis = {
            'modality_quality': {
                'text': text_info['quality_score'].mean().item(),
                'audio': audio_info['quality_score'].mean().item(), 
                'vision': vision_info['quality_score'].mean().item()
            },
            'fusion_weights': {
                'text_alpha': text_info['alpha'],
                'audio_alpha': audio_info['alpha'],
                'vision_alpha': vision_info['alpha']
            }
        }
        
        return fused_features, attention_analysis
