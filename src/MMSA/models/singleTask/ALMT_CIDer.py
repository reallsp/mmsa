"""
ALMT_CIDer: 多模态情感分析模型

基于ALMT架构的增强版本，支持文本、音频、视频三模态融合。
核心特性：
- 原始ALMT骨干网络
- 双向交叉注意力机制
- 智能特征对齐 （弃用-没有效果，因为原始的模块已经提供相关对齐功能）
- CIDer知识蒸馏
- 模态贡献平衡

支持模块化配置，可选择启用/关闭特定增强功能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
import yaml
import os
from ..subNets import BertTextEncoder
from .ALMT import *

class ALMT_CIDer(nn.Module):
    
    def __init__(self, args):
        super(ALMT_CIDer, self).__init__()
        self.args = args
        
        # 加载配置
        self.config = self._load_config()
        self._update_args_from_config()
        
        self._init_original_almt_core()
        self._init_enhancement_layers()
        self._print_architecture_info()
    
    def _init_original_almt_core(self):
        # 从配置中获取ALMT参数
        almt_config = self.config['almt_config']
        
        # 模态特征参数
        self.orig_d_l, self.orig_d_a, self.orig_d_v = almt_config['feature_dims']
        self.orig_length_l, self.orig_length_a, self.orig_length_v = almt_config['feature_length']
        self.dst_embedding_d = almt_config['dst_feature_dims']
        self.dst_embedding_hidden_d = almt_config['dst_feature_hidden_dims']
        self.dst_embedding_length = almt_config['dst_embedding_length']
        self.embedding_depth_l, self.embedding_depth_a, self.embedding_depth_v = almt_config['embedding_depth']
        self.embedding_heads_l, self.embedding_heads_a, self.embedding_heads_v = almt_config['embedding_heads']
        
        # 超参数学习参数
        self.h_hyper_d = almt_config['dst_feature_dims']
        self.h_hyper_length = almt_config['dst_embedding_length']
        self.AHL_depth = almt_config['AHL_depth']
        self.h_hyper_layer_heads = almt_config['h_hyper_layer_heads']
        self.l_encoder_heads = almt_config['l_encoder_heads']
        
        # 融合层参数
        self.fusion_d = self.h_hyper_d
        self.fusion_hidden_d = almt_config['fusion_hidden_d']
        self.fusion_heads = almt_config['fusion_heads']
        self.fusion_layer_depth = almt_config['fusion_layer_depth']
        
        # BERT编码器
        if almt_config['use_bert']:
            self.bertmodel = BertTextEncoder(
                use_finetune=almt_config['use_finetune'],
                transformers=almt_config['transformers'],
                pretrained=almt_config['pretrained']
            )
        self.use_bert = almt_config['use_bert']
        
        # 模态嵌入层
        self.embedding_l = nn.Sequential(
            nn.Linear(self.orig_d_l, self.dst_embedding_d),
            Transformer(
                num_frames=self.orig_length_l,
                save_hidden=False,
                token_len=self.dst_embedding_length,
                dim=self.dst_embedding_d,
                depth=self.embedding_depth_l,
                heads=self.embedding_heads_l,
                mlp_dim=self.dst_embedding_hidden_d
            )
        )
        
        self.embedding_a = nn.Sequential(
            nn.Linear(self.orig_d_a, self.dst_embedding_d),
            Transformer(
                num_frames=self.orig_length_a,
                save_hidden=False,
                token_len=self.dst_embedding_length,
                dim=self.dst_embedding_d,
                depth=self.embedding_depth_a,
                heads=self.embedding_heads_a,
                mlp_dim=self.dst_embedding_hidden_d
            )
        )
        
        self.embedding_v = nn.Sequential(
            nn.Linear(self.orig_d_v, self.dst_embedding_d),
            Transformer(
                num_frames=self.orig_length_v,
                save_hidden=False,
                token_len=self.dst_embedding_length,
                dim=self.dst_embedding_d,
                depth=self.embedding_depth_v,
                heads=self.embedding_heads_v,
                mlp_dim=self.dst_embedding_hidden_d
            )
        )
        
        # 超参数学习层
        self.h_hyper = nn.Parameter(torch.ones(1, self.h_hyper_length, self.h_hyper_d))
        
        self.l_encoder = Transformer(
            num_frames=self.dst_embedding_length,
            save_hidden=True,
            token_len=None,
            dim=self.dst_embedding_d,
            depth=self.AHL_depth-1,
            heads=self.l_encoder_heads,
            mlp_dim=self.dst_embedding_hidden_d
        )
        
        self.h_hyper_layer = HhyperLearningEncoder(
            dim=self.h_hyper_d,
            depth=self.AHL_depth,
            heads=self.h_hyper_layer_heads,
            dim_head=int(self.h_hyper_d/self.h_hyper_layer_heads)
        )
        
        # 融合层和回归头
        self.fusion_layer = CrossTransformer(
            source_num_frames=self.dst_embedding_length,
            tgt_num_frames=self.dst_embedding_length,
            dim=self.fusion_d,
            depth=self.fusion_layer_depth,
            heads=self.fusion_heads,
            mlp_dim=self.fusion_hidden_d
        )
        
        self.regression_head = nn.Linear(128, 1)
        
        # 模块开关配置
        self.enable_cider_distill = self.config.get('enable_cider_distillation', False)
        self.enable_alignment = self.config.get('enable_intelligent_alignment', False)
        self.enable_contribution_balance = self.config.get('enable_modality_balance', False)
        self.enable_bidirectional_attn = self.config.get('enable_bidirectional_attention', False)
    
    def _init_enhancement_layers(self):
        """初始化增强模块"""
        if self.enable_bidirectional_attn:
            self._init_bidirectional_cross_attention_layer()
        
        if self.enable_alignment:
            self._init_intelligent_alignment_layer()
        
        if self.enable_cider_distill:
            self._init_cider_distillation_layer()
        
        if self.enable_contribution_balance:
            self._init_modality_balance_layer()
    
    def _init_bidirectional_cross_attention_layer(self):
        """初始化双向交叉注意力层"""
        bidirectional_config = self.config.get('bidirectional_cross_attention', {})
        
        self.cross_modal_attention = BidirectionalCrossModalAttention(
            dim=self.dst_embedding_d,
            heads=bidirectional_config.get('attention_heads', 8),
            dropout=bidirectional_config.get('attention_dropout', 0.1),
            temperature=bidirectional_config.get('temperature', 1.2),
            alpha=bidirectional_config.get('complementary_alpha', 0.6),
            use_complementary=bidirectional_config.get('use_complementary', True)
        )
        
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(self.dst_embedding_d * 3, self.dst_embedding_d * 2),
            nn.LayerNorm(self.dst_embedding_d * 2),
            nn.ReLU(),
            nn.Dropout(bidirectional_config.get('fusion_dropout', 0.1)),
            nn.Linear(self.dst_embedding_d * 2, self.dst_embedding_d),
            nn.LayerNorm(self.dst_embedding_d)
        )
        
        self.cross_attn_residual_weight = nn.Parameter(
            torch.tensor(bidirectional_config.get('residual_weight', 0.2))
        )
    
    def _init_intelligent_alignment_layer(self):
        """初始化智能对齐层"""
        alignment_config = self.config['intelligent_alignment']
        
        self.importance_scorer = nn.Sequential(
            nn.Linear(self.dst_embedding_d, self.dst_embedding_d // 2),
            nn.ReLU(),
            nn.Dropout(alignment_config['alignment_dropout']),
            nn.Linear(self.dst_embedding_d // 2, 1),
            nn.Sigmoid()
        )
        
        self.alignment_quality_estimator = nn.Sequential(
            nn.Linear(self.dst_embedding_d, 1),
            nn.Sigmoid()
        )
    
    def _init_cider_distillation_layer(self):
        """初始化CIDer蒸馏层"""
        distill_config = self.config['cider_self_distillation']
        
        self.h_hyper_conservative = nn.Parameter(
            torch.ones(1, self.h_hyper_length, self.h_hyper_d) * distill_config['conservative_init_scale']
        )
        
        self.h_hyper_adaptive = nn.Parameter(
            torch.ones(1, self.h_hyper_length, self.h_hyper_d) * distill_config['adaptive_init_scale']
        )
    
    def _init_modality_balance_layer(self):
        """初始化模态平衡层"""
        balance_config = self.config['modality_contribution_balance']
        
        self.modality_experts = nn.ModuleDict({
            'text': ModalityExpert(self.fusion_d, balance_config['expert_dropout']),
            'audio': ModalityExpert(self.fusion_d, balance_config['expert_dropout']),
            'video': ModalityExpert(self.fusion_d, balance_config['expert_dropout'])
        })
        
        self.contribution_estimator = nn.Sequential(
            nn.Linear(self.fusion_d, self.fusion_d // 2),
            nn.ReLU(),
            nn.Linear(self.fusion_d // 2, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, text, audio, video, masks=None, text_lengths=None, labels=None, training=True):
        """前向传播"""
        if self._is_baseline_mode():
            return self._original_almt_forward(text, audio, video)
        else:
            return self._enhanced_forward(text, audio, video, masks, text_lengths, labels, training)
    
    def _is_baseline_mode(self):
        """检查是否为baseline模式"""
        return not any([
            self.enable_cider_distill,
            self.enable_alignment,
            self.enable_contribution_balance,
            self.enable_bidirectional_attn
        ])
    
    def _original_almt_forward(self, text, audio, video):
        """原始ALMT前向传播"""
        batch_size = video.size(0)
        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=batch_size)

        # 文本编码
        if self.use_bert:
            x_text = self.bertmodel(text)
        else:
            x_text = text

        # 模态嵌入
        h_l = self.embedding_l(x_text)[:, :8]
        h_a = self.embedding_a(audio)[:, :8]
        h_v = self.embedding_v(video)[:, :8]

        # AHL
        h_l_list = self.l_encoder(h_l)
        h_hyper = self.h_hyper_layer(h_l_list, h_a, h_v, h_hyper)
        
        # 融合
        feat = self.fusion_layer(h_hyper, h_l_list[-1])[:, 0]

        # 回归
        output = self.regression_head(feat)
        return output
    
    def _enhanced_forward(self, text, audio, video, masks, text_lengths, labels, training):
        """增强前向传播"""
        # 原始ALMT特征提取
        base_features = self._extract_original_almt_features(text, audio, video)
        current_features = base_features.copy()
        
        losses = {}
        
        # 双向交叉注意力增强
        if self.enable_bidirectional_attn:
            current_features, cross_attn_loss = self._apply_bidirectional_cross_attention(
                current_features, masks, text_lengths
            )
            losses['cross_attention_loss'] = cross_attn_loss
        
        # 智能特征对齐
        if self.enable_alignment:
            current_features, alignment_loss = self._apply_intelligent_alignment(
                current_features, masks, text_lengths
            )
            losses['alignment_loss'] = alignment_loss
        
        # CIDer知识蒸馏
        if self.enable_cider_distill and training:
            final_result, distillation_loss = self._apply_cider_distillation(
                current_features, labels
            )
            losses['distillation_loss'] = distillation_loss
        else:
            final_result = self._standard_forward_with_features(current_features)
        
        # 模态贡献平衡
        if self.enable_contribution_balance:
            final_result, contribution_loss = self._apply_modality_balance(
                final_result, current_features, labels
            )
            losses['contribution_loss'] = contribution_loss
        
        # 组织返回结果
        if training and losses:
            return {
                'prediction': final_result,
                **{k: v for k, v in losses.items()},
                **{k: torch.tensor(0.0).to(video.device) for k in 
                   ['distillation_loss', 'alignment_loss', 'contribution_loss', 'cross_attention_loss'] 
                   if k not in losses}
            }
        else:
            return final_result
    
    def _extract_original_almt_features(self, text, audio, video):
        """提取原始ALMT特征"""
        if self.use_bert:
            x_text = self.bertmodel(text)
        else:
            x_text = text
        
        h_l = self.embedding_l(x_text)[:, :8]
        h_a = self.embedding_a(audio)[:, :8]
        h_v = self.embedding_v(video)[:, :8]
        h_l_list = self.l_encoder(h_l)
        
        return {
            'h_l': h_l,
            'h_a': h_a,
            'h_v': h_v,
            'h_l_list': h_l_list,
            'x_text': x_text
        }
    
    def _apply_bidirectional_cross_attention(self, features, masks, text_lengths):
        """应用双向交叉注意力增强"""
        h_l, h_a, h_v = features['h_l'], features['h_a'], features['h_v']
        
        # 双向交叉注意力计算
        cross_attn_outputs = self.cross_modal_attention(h_l, h_a, h_v)
        
        # 融合跨模态特征
        enhanced_features = torch.cat([
            cross_attn_outputs['text_enhanced'],
            cross_attn_outputs['audio_enhanced'], 
            cross_attn_outputs['video_enhanced']
        ], dim=-1)
        
        fused_features = self.cross_modal_fusion(enhanced_features)
        
        # 残差连接
        residual_weight = torch.sigmoid(self.cross_attn_residual_weight)
        h_l_enhanced = h_l * (1 - residual_weight) + fused_features * residual_weight
        h_a_enhanced = h_a * (1 - residual_weight) + fused_features * residual_weight
        h_v_enhanced = h_v * (1 - residual_weight) + fused_features * residual_weight
        
        # 计算交叉注意力损失
        cross_attn_loss = self._compute_cross_attention_loss(cross_attn_outputs)
        
        # 更新特征
        enhanced_features = features.copy()
        enhanced_features.update({
            'h_l': h_l_enhanced,
            'h_a': h_a_enhanced,
            'h_v': h_v_enhanced
        })
        
        return enhanced_features, cross_attn_loss
    
    def _apply_intelligent_alignment(self, features, masks, text_lengths):
        """应用智能特征对齐"""
        h_l = features['h_l']
        h_a = features['h_a']
        h_v = features['h_v']
        
        # 基于重要性的对齐
        text_importance = self.importance_scorer(h_l).squeeze(-1)
        _, top_indices = torch.topk(text_importance, k=8, dim=1)
        batch_indices = torch.arange(h_l.size(0)).unsqueeze(1).expand(-1, 8)
        
        # 对齐音频和视频特征
        aligned_h_a = h_a[batch_indices, top_indices]
        aligned_h_v = h_v[batch_indices, top_indices]
        
        # 计算对齐损失
        alignment_loss = self._compute_alignment_loss(h_l, aligned_h_a, aligned_h_v, text_importance, top_indices)
        
        # 更新特征
        aligned_features = features.copy()
        aligned_features.update({
            'h_a': aligned_h_a,
            'h_v': aligned_h_v
        })
        
        return aligned_features, alignment_loss
    
    def _apply_cider_distillation(self, features, labels):
        """应用CIDer知识蒸馏"""
        batch_size = features['h_l'].size(0)
        
        # 保守策略
        h_hyper_conservative = repeat(self.h_hyper_conservative, '1 n d -> b n d', b=batch_size)
        conservative_result = self._almt_forward_with_hyper(features, h_hyper_conservative, batch_size)
        
        # 激进策略
        h_hyper_adaptive = repeat(self.h_hyper_adaptive, '1 n d -> b n d', b=batch_size)
        adaptive_result = self._almt_forward_with_hyper(features, h_hyper_adaptive, batch_size)
        
        # 计算蒸馏损失
        distillation_loss = self._compute_distillation_loss(conservative_result, adaptive_result)
        
        # 选择最终策略
        final_prediction = conservative_result['prediction']
        
        return final_prediction, distillation_loss
    
    def _apply_modality_balance(self, prediction, features, labels):
        """应用模态贡献平衡"""
        contribution_loss = torch.tensor(0.0).to(prediction.device)
        return prediction, contribution_loss
    
    def _standard_forward_with_features(self, features):
        """使用增强特征的标准前向传播"""
        batch_size = features['h_l'].size(0)
        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=batch_size)
        
        # 使用增强后的特征继续ALMT流程
        h_hyper = self.h_hyper_layer(features['h_l_list'], features['h_a'], features['h_v'], h_hyper)
        feat = self.fusion_layer(h_hyper, features['h_l_list'][-1])[:, 0]
        output = self.regression_head(feat)
        
        return output
    
    def _almt_forward_with_hyper(self, features, h_hyper_param, batch_size):
        """使用指定h_hyper参数的ALMT前向传播"""
        h_hyper = self.h_hyper_layer(features['h_l_list'], features['h_a'], features['h_v'], h_hyper_param)
        feat = self.fusion_layer(h_hyper, features['h_l_list'][-1])[:, 0]
        prediction = self.regression_head(feat)
        
        return {
            'prediction': prediction,
            'h_hyper': h_hyper,
            'feat': feat
        }
    
    def _compute_cross_attention_loss(self, cross_attn_outputs):
        """计算交叉注意力损失"""
        return torch.tensor(0.0).to(list(cross_attn_outputs.values())[0].device)
    
    def _compute_alignment_loss(self, h_l, aligned_h_a, aligned_h_v, text_importance, top_indices):
        """计算对齐损失"""
        text_selected = h_l[torch.arange(h_l.size(0)).unsqueeze(1), top_indices]
        cos_sim_a = F.cosine_similarity(text_selected, aligned_h_a, dim=-1)
        cos_sim_v = F.cosine_similarity(text_selected, aligned_h_v, dim=-1)
        alignment_loss = 1.0 - (cos_sim_a.mean() + cos_sim_v.mean()) / 2
        
        return alignment_loss
    
    def _compute_distillation_loss(self, conservative_results, adaptive_results):
        """计算蒸馏损失"""
        # 预测蒸馏损失
        prediction_loss = F.mse_loss(
            conservative_results['prediction'], 
            adaptive_results['prediction']
        )
        
        # 特征蒸馏损失
        feature_loss = F.mse_loss(
            conservative_results['feat'],
            adaptive_results['feat']
        )
        
        # h_hyper蒸馏损失
        hyper_loss = F.mse_loss(
            conservative_results['h_hyper'],
            adaptive_results['h_hyper']
        )
        
        # 加权组合
        distill_weights = self.config['cider_self_distillation']['distillation_weights']
        total_loss = (
            prediction_loss * distill_weights['prediction_distill'] +
            feature_loss * distill_weights['feature_distill'] +
            hyper_loss * distill_weights['hyper_distill']
        )
        
        return total_loss
    
    def _load_config(self):
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), "../../config/almt_cider.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.args._yaml_config = config
        return config
    
    def _update_args_from_config(self):
        """更新args配置"""
        config = self.config
        
        if 'training_config' in config:
            training_config = config['training_config']
            self.args.max_epochs = training_config.get('max_epochs', getattr(self.args, 'max_epochs', 200))
            self.args.batch_size = training_config.get('batch_size', getattr(self.args, 'batch_size', 64))
            self.args.learning_rate = training_config.get('learning_rate', getattr(self.args, 'learning_rate', 0.0001))
            self.args.weight_decay = training_config.get('weight_decay', getattr(self.args, 'weight_decay', 0.0001))
            self.args.early_stop = training_config.get('early_stop', getattr(self.args, 'early_stop', 32))
            self.args.max_grad_norm = training_config.get('max_grad_norm', getattr(self.args, 'max_grad_norm', 2.0))
    
    def _print_architecture_info(self):
        """打印架构信息"""
        print("ALMT_CIDer 架构已初始化")
        print(f"  Baseline模式: {'是' if self._is_baseline_mode() else '否'}")
        if not self._is_baseline_mode():
            print(f"  启用模块:")
            if self.enable_bidirectional_attn:
                print(f"    - 双向交叉注意力")
            if self.enable_alignment:
                print(f"    - 智能特征对齐")
            if self.enable_cider_distill:
                print(f"    - CIDer知识蒸馏")
            if self.enable_contribution_balance:
                print(f"    - 模态贡献平衡")


class BidirectionalCrossModalAttention(nn.Module):
    """双向交叉模态注意力模块"""
    
    def __init__(self, dim, heads=8, dropout=0.1, temperature=1.0, alpha=0.6, use_complementary=True):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.temperature = temperature
        self.alpha = alpha
        self.use_complementary = use_complementary
        
        # 注意力投影层
        self.to_q_text = nn.Linear(dim, dim, bias=False)
        self.to_k_text = nn.Linear(dim, dim, bias=False)
        self.to_v_text = nn.Linear(dim, dim, bias=False)
        
        self.to_q_audio = nn.Linear(dim, dim, bias=False)
        self.to_k_audio = nn.Linear(dim, dim, bias=False)
        self.to_v_audio = nn.Linear(dim, dim, bias=False)
        
        self.to_q_video = nn.Linear(dim, dim, bias=False)
        self.to_k_video = nn.Linear(dim, dim, bias=False)
        self.to_v_video = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, h_text, h_audio, h_video):
        batch_size, seq_len, dim = h_text.shape
        
        # 计算查询、键、值
        q_t, k_t, v_t = self.to_q_text(h_text), self.to_k_text(h_text), self.to_v_text(h_text)
        q_a, k_a, v_a = self.to_q_audio(h_audio), self.to_k_audio(h_audio), self.to_v_audio(h_audio)
        q_v, k_v, v_v = self.to_q_video(h_video), self.to_k_video(h_video), self.to_v_video(h_video)
        
        # 重塑为多头注意力格式
        q_t, k_t, v_t = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.heads), (q_t, k_t, v_t))
        q_a, k_a, v_a = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.heads), (q_a, k_a, v_a))
        q_v, k_v, v_v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.heads), (q_v, k_v, v_v))
        
        # 跨模态注意力计算
        # Text -> Audio/Video
        attn_ta = torch.matmul(q_t, k_a.transpose(-2, -1)) * self.scale / self.temperature
        attn_tv = torch.matmul(q_t, k_v.transpose(-2, -1)) * self.scale / self.temperature
        
        # Audio -> Text/Video  
        attn_at = torch.matmul(q_a, k_t.transpose(-2, -1)) * self.scale / self.temperature
        attn_av = torch.matmul(q_a, k_v.transpose(-2, -1)) * self.scale / self.temperature
        
        # Video -> Text/Audio
        attn_vt = torch.matmul(q_v, k_t.transpose(-2, -1)) * self.scale / self.temperature
        attn_va = torch.matmul(q_v, k_a.transpose(-2, -1)) * self.scale / self.temperature
        
        # 应用softmax
        attn_ta, attn_tv = F.softmax(attn_ta, dim=-1), F.softmax(attn_tv, dim=-1)
        attn_at, attn_av = F.softmax(attn_at, dim=-1), F.softmax(attn_av, dim=-1)
        attn_vt, attn_va = F.softmax(attn_vt, dim=-1), F.softmax(attn_va, dim=-1)
        
        # 计算增强特征
        text_enhanced = torch.matmul(attn_ta, v_a) + torch.matmul(attn_tv, v_v)
        audio_enhanced = torch.matmul(attn_at, v_t) + torch.matmul(attn_av, v_v)
        video_enhanced = torch.matmul(attn_vt, v_t) + torch.matmul(attn_va, v_a)
        
        # 重塑回原始维度
        text_enhanced = rearrange(text_enhanced, 'b h n d -> b n (h d)')
        audio_enhanced = rearrange(audio_enhanced, 'b h n d -> b n (h d)')
        video_enhanced = rearrange(video_enhanced, 'b h n d -> b n (h d)')
        
        # 输出投影
        text_enhanced = self.out_proj(text_enhanced)
        audio_enhanced = self.out_proj(audio_enhanced)
        video_enhanced = self.out_proj(video_enhanced)
        
        return {
            'text_enhanced': text_enhanced,
            'audio_enhanced': audio_enhanced,
            'video_enhanced': video_enhanced
        }


class ModalityExpert(nn.Module):
    """模态专家网络"""
    
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.expert = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        return self.expert(x)
