"""
From: https://github.com/thuiar/Self-MM
Paper: Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis
"""
# self supervised multimodal multi-task learning network
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
try:
    from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv
except ImportError:
    # torch_geometric is optional (graph_enhance is not used in forward pass)
    GCNConv = GATv2Conv = SAGEConv = None
from ..subNets.BertTextEncoder import BertTextEncoder
import math

__all__ = ['GMM']



import torch
try:
    import pywt
except ImportError:
    # pywt is optional (wavelet transform is not used in forward pass)
    pywt = None
import numpy as np

import torch.nn as nn
import torch.utils.checkpoint as checkpoint
try:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    # timm is optional - provide fallback implementations
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output
    def to_2tuple(x):
        if isinstance(x, (list, tuple)):
            return x
        return (x, x)
    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        # Simple fallback - just use normal initialization
        nn.init.normal_(tensor, mean=mean, std=std)




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias) 
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.q_poe = nn.Conv1d(dim, dim, 3, padding=1, groups=num_heads)
        self.v_poe = nn.Conv1d(dim, dim, 3, padding=1, groups=num_heads)

    def forward(self, x_q, x_kv):
        """
        Args:
            x_q: input features with shape of (B, N, C)
            x_kv: input features with shape of (B, N, C)
        """
        x = x_kv
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        # 生成k,v
        kv = self.kv(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        # 生成q
        q = self.q(x_q).reshape(b, n, c)

        # 非线性变换
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        # 多头注意力
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # (b, num_heads, n, head_dim)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        # 线性注意力计算
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q @ kv * z

        # 重塑并添加位置编码
        x = x.transpose(1, 2).reshape(b, n, c)  # (b, n, c)
        v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1)  # (b, c, n)
        x = x + self.lepe(v).permute(0, 2, 1)  # (b, n, c)

        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
    
class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim

        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x_q,x_kv):
        # Input: x_q and x_kv are (B, L, C)
        # Keep both as (B, L, C) for LinearAttention which expects (B, N, C)
        B, L, C = x_kv.shape

        # Use permute for Conv1d (needs (B, C, L))
        # x_kv is (B, L, C), permute to (B, C, L) for Conv1d
        cpe1_out = self.cpe1(x_kv.permute(0, 2, 1))  # (B, L, C) -> (B, C, L) -> Conv1d -> (B, C, L)
        x = x_kv + cpe1_out.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        shortcut = x
        # 

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, C, L)  # (B, L, C) -> (B, C, L)
        x = self.act(self.dwc(x)).view(B, L, C)  # (B, C, L) -> (B, L, C)

        # Linear Attention: both x_q and x should be (B, L, C)
        x = self.attn(x_q, x)
        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        # Final CPE: x is (B, L, C), permute to (B, C, L) for Conv1d
        cpe2_out = self.cpe2(x.permute(0, 2, 1))  # (B, L, C) -> (B, C, L) -> Conv1d -> (B, C, L)
        x = x + cpe2_out.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # Return as (B, L, C) - no need to permute, already in correct format
        return x



class GMM(nn.Module):
    def __init__(self, args):
        super(GMM, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned


        # BERT编码器用于文本特征提取
        self.text_model = BertTextEncoder(
            use_finetune=getattr(args, 'use_finetune', False),
            transformers=getattr(args, 'transformers', 'bert'),
            pretrained=getattr(args, 'pretrained', 'bert-base-uncased'),
        )
        # 文本特征维度 - use text_out from args (should be divisible by num_heads)
        text_out = getattr(args, 'text_out', 768)
        # Ensure text_out is divisible by num_heads (5) for MLLABlock
        # Round down to nearest multiple of 5
        self.text_dim = (text_out // 5) * 5 if text_out % 5 != 0 else text_out
        
        # 文本到视频的多头注意力模块
        self.lca_t2v = MLLABlock(self.text_dim, num_heads=5)
        # 文本到音频的多头注意力模块 
        self.lca_t2a = MLLABlock(self.text_dim, num_heads=5)
        # Projection layers to match text_dim for audio_share and video_share
        self.audio_share_proj_to_textdim = nn.Linear(args.text_out, self.text_dim)
        self.video_share_proj_to_textdim = nn.Linear(args.text_out, self.text_dim)
        # Projection back from text_dim to text_out
        self.audio_share_proj_back = nn.Linear(self.text_dim, args.text_out)
        self.video_share_proj_back = nn.Linear(self.text_dim, args.text_out)
        
        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_linear = nn.Linear(audio_in, args.text_out*2)
        self.video_linear = nn.Linear(video_in, args.text_out*2)
        # fusion subnets
        self.audio_model = AuViSubNet(args.text_out, args.a_lstm_hidden_size, args.text_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(args.text_out, args.v_lstm_hidden_size, args.text_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)
        self.graph_enhance = graph_enhance(args.text_out)
        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(args.text_out *3, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)
        # the post_fusion layers
        self.post_fusion_dropout_ = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1_ = nn.Linear(args.text_out *3, args.post_fusion_dim)
        self.post_fusion_layer_2_ = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3_ = nn.Linear(args.post_fusion_dim, 1)
        # the classify layer for text
        self.a_d_text = nn.Linear(args.text_out, args.text_out*2)
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.a_d_audio = nn.Linear(args.text_out, args.text_out*2)
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.text_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.a_d_video = nn.Linear(args.text_out, args.text_out*2)
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.text_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

    def forward(self,text, audio, video):
        # Unpack audio and video tuples (they are always passed as tuples)
        if isinstance(audio, tuple):
            audio, audio_lengths = audio
        else:
            audio_lengths = None
        if isinstance(video, tuple):
            video, video_lengths = video
        else:
            video_lengths = None
        
        # Apply linear transformation
        audio = self.audio_linear(audio)
        video = self.video_linear(video)
        audio_share,audio_pri = torch.chunk(audio, 2, dim=-1)
        video_share,video_pri = torch.chunk(video, 2, dim=-1)
        # audio_share = self.lca_t2a(text,audio_share)
        # video_share = self.lca_t2v(text,video_share)
        # audio_fusion = self.lca_t2a(text,audio_fusion)
        # video_fusion = self.lca_t2v(text,video_fusion)sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        text = self.text_model(text)
        # video = self.lca_t2v(video,video)
        # audio = self.lca_t2a(audio,audio)
        
        

        if self.aligned:
            # Project audio_share and video_share to text_dim for MLLABlock
            audio_share_proj = self.audio_share_proj_to_textdim(audio_share)
            video_share_proj = self.video_share_proj_to_textdim(video_share)
            # Apply MLLABlock
            audio_share_proj = self.lca_t2a(audio_share_proj, audio_share_proj)
            video_share_proj = self.lca_t2v(video_share_proj, video_share_proj)
            # Project back to text_out
            audio_share = self.audio_share_proj_back(audio_share_proj)
            video_share = self.video_share_proj_back(video_share_proj)
            
            text = text[:,0,:]
            # For aligned mode, use text_lengths
            audio_share = self.audio_model(audio_share, text_lengths)
            video_share = self.video_model(video_share, text_lengths)
            audio_pri = self.audio_model(audio_pri, text_lengths)
            video_pri = self.video_model(video_pri, text_lengths)
        else:
            # For unaligned mode, use audio_lengths and video_lengths
            audio_share = self.audio_model(audio_share, audio_lengths)
            video_share = self.video_model(video_share, video_lengths)
            audio_pri = self.audio_model(audio_pri, audio_lengths)
            video_pri = self.video_model(video_pri, video_lengths)
        
        # text_a,text_d = haar_wavelet_transform(text)
        # audio_a,audio_d = haar_wavelet_transform(audio)
        # video_a,video_d = haar_wavelet_transform(video)

        # text = text_a
        # audio = audio_a
        # video = video_a
        # fusion
        fusion_h = torch.cat([text, audio_share, video_share], dim=-1)
        # fusion_h = self.graph_enhance(fusion_h)
        # fusion==>text, audio, vision 
        # Split concatenated features back for separate processing
        fusion_text, fusion_audio, fusion_video = torch.chunk(fusion_h, 3, dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # text = text_d
        # audio = audio_d
        # video = video_d
        # # text
        audio = audio_pri
        video = video_pri
        # fusion = torch.cat([text, audio, video], dim=-1)
        fusion = torch.cat([fusion_text, fusion_audio, fusion_video], dim=-1)
        fusion = self.post_fusion_dropout(fusion)
        fusion = F.relu(self.post_fusion_layer_1(fusion), inplace=False)
        # fusion = self.post_fusion_layer_2(fusion)
        # fusion = self.post_fusion_layer_3(fusion)
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion,
            'Fusion_text': fusion_text,
            'Fusion_audio': fusion_audio,
            'Fusion_video': fusion_video,
        }
        return res

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        setattr(self.weight, 'gcn_weight', True)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            setattr(self.bias, 'gcn_weight', True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def adjacency_matrix_to_edge_index(adj_matrix):
    # 将邻接矩阵转换为边索引
    indices = torch.where(adj_matrix != 0)
    edge_index = torch.stack((indices[0], indices[1]), dim=0)
    
    # 使用 PyTorch 的多次排序实现多键排序
    sorted_indices = torch.argsort(edge_index[1], stable=True)
    edge_index = edge_index[:, sorted_indices]
    sorted_indices = torch.argsort(edge_index[0], stable=True)
    edge_index = edge_index[:, sorted_indices]
    
    return edge_index

class CADM(nn.Module):
    def __init__(self, feature_dim,alpha=0.9):
        super(CADM, self).__init__()
        self.feature_dim = feature_dim
        self.alpha=alpha
        self.Uq = nn.Linear(feature_dim, feature_dim)  # 可学习的投影矩阵
        self.Ukv = nn.Linear(feature_dim, 2 * feature_dim)  # 可学习的投影矩阵
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),  # 输入特征和领域风格中心的维度之和
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.pre_phi = None

    def forward(self, features,num_domains):
        #[B*num_domains,D]
        domain_num = num_domains
        batch_domain, feature_dim = features.size()
        batch_size = batch_domain//domain_num
        features = features.view(batch_size, domain_num, feature_dim) 
        # 在每个domain上计算均值
        f = torch.mean(features,dim=0)

        Q = self.Uq(f.view(domain_num,feature_dim))
        feature = features.view(batch_size*domain_num, feature_dim)
        KV = self.Ukv(feature).chunk(2, dim=-1)
        K,V = KV[0],KV[1]
        # Q:[B,domain_num,D] K:[B*domain,D]==>[B,domain_num,D]
        attention_maps = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) / (self.feature_dim ** 0.5), dim=-1)  # [batch_size*num_domains, 1, batch_size*num_domains]
        contrary_attention_maps = (1-attention_maps)/torch.sum(1-attention_maps)
        phi_g = torch.matmul(contrary_attention_maps, V.view(batch_size*domain_num, feature_dim))
        # if self.pre_phi is not None:
        #     phi_g = self.alpha*self.pre_phi + (1-self.alpha)*phi_g
        # else:
        #     phi_g = phi_g
        #     self.pre_phi = phi_g
        fused_features = self.mlp(torch.cat([features, phi_g.unsqueeze(0).expand(batch_size, domain_num, feature_dim)], dim=-1))  
        fused_features = fused_features.view(batch_size*domain_num, feature_dim)
        return fused_features

class GCN_2ADJ(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN_2ADJ, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid*4)
        self.gc2 = GraphConvolution(nhid*4, nclass)
        # self.gc1 = GCNConv(nfeat, nhid*4)
        # self.gc2 = GCNConv(nhid*4, nclass)
        # self.gc1 = SAGEConv(nfeat, nhid,aggr = 'lstm')
        # self.gc2 = SAGEConv(nhid, nclass,aggr = 'lstm')
        # self.gc1 = GATv2Conv(nfeat, nhid, heads=4,concat=True)
        # self.gc2 = GATv2Conv(nhid*4, nclass, heads=4,concat=False)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj_1, adj_2):
        # adj_1 = adjacency_matrix_to_edge_index(adj_1)
        # adj_2 = adjacency_matrix_to_edge_index(adj_2)
        # adj_1, _ = adj_1.sort(1)
        # adj_2, _ = adj_2.sort(1)
        # identity = x 
        # x = self.gc1(x, edge_index=edge_index1)+identity
        # x = self.relu(x)
        # identity = x
        # x = self.gc2(x, edge_index=edge_index2)+identity
        x = self.gc1(x, adj_1.detach())
        x = self.relu(x)
        x = self.gc2(x, adj_2.detach())

        return x



import torch
import torch.nn as nn

def adjacency_matrix_to_edge_index(adj_matrix):
    """将邻接矩阵转换为边索引"""
    indices = torch.where(adj_matrix != 0)
    edge_index = torch.stack((indices[0], indices[1]), dim=0)
    return edge_index

class ADJ_FirstLayer(nn.Module):
    def __init__(self):
        super(ADJ_FirstLayer, self).__init__()

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def forward(self, x):
        N, C = x.size()
        
        # 初始化邻接矩阵
        adj = torch.eye(N + 3).cuda()
        adj[N:, N:] = 1.0

        # 生成归一化邻接矩阵
        adj = self.gen_adj(adj)

        # 转换为边索引
        return adj


class ADJ_AttCenter(nn.Module):
    def __init__(self, dim):
        super(ADJ_AttCenter, self).__init__()
        self.attention = nn.Linear(dim, 1)

        self.register_buffer('running_mean', torch.zeros(3, dim))
        self.momentum = 0.0

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def forward(self, x, mode='train'):
        N, C = x.size()
        if self.attention.weight.requires_grad:
            mode='train'
        else:
            mode='test'
        # mode = "test"
        if mode == 'train':
            stride = N // 3
            weight = self.attention(x)  # (N, 1)
            weight0, weight1, weight2 = (
                weight[:stride], 
                weight[stride: 2 * stride], 
                weight[2 * stride:]
            )
            
            weight0 = weight0 / weight0.sum(dim=0, keepdim=False)
            weight1 = weight1 / weight1.sum(dim=0, keepdim=False)
            weight2 = weight2 / weight2.sum(dim=0, keepdim=False)

            center0 = torch.sum(x[:stride] * weight0, dim=0, keepdim=True)
            center1 = torch.sum(x[stride: 2 * stride] * weight1, dim=0, keepdim=True)
            center1 = (center1 + center0) / 2.0
            center2 = torch.sum(x[2 * stride:] * weight2, dim=0, keepdim=True)
            center2 = (center2 + center1) / 2.0
            center_mean = (center0 + center1 + center2) / 3.0

            x_new = torch.cat([x, center0, center0, center0], dim=0)
            # 更新运行时均值
            self.running_mean.mul_(self.momentum)
            self.running_mean[0].add_((1 - self.momentum) * center0[0].data)
            self.running_mean[1].add_((1 - self.momentum) * center1[0].data)
            self.running_mean[2].add_((1 - self.momentum) * center2[0].data)

            adj = torch.eye(N + 3).cuda()
            adj[N:, N:] = 1.0

            # 构造邻接矩阵
            adj[:stride, N] = 1.0
            adj[N, :stride] = 1.0
            adj[stride: 2 * stride, N + 1] = 1.0
            adj[N + 1, stride: 2 * stride] = 1.0
            adj[2 * stride: 3 * stride, N + 2] = 1.0
            adj[N + 2, 2 * stride:] = 1.0

        else:
            x_new = torch.cat([x, torch.autograd.Variable(self.running_mean)], dim=0)
            adj = torch.eye(N + 3).cuda()
            adj[N:, N:] = 1.0
            adj[:N, N] = 1.0
            adj[N, :N] = 1.0

        # 生成归一化邻接矩阵
        adj = self.gen_adj(adj)

        
        return x_new, adj
    
class graph_enhance(nn.Module):
    def __init__(self, dim):
        super(graph_enhance, self).__init__()
        self.adj1 = ADJ_FirstLayer()
        self.adj2 = ADJ_AttCenter(dim)
        self.gcn = GCN_2ADJ(dim, dim, dim)
    def forward(self, x,mode='test'):
        adj_1 = self.adj1(x)
        x_new, adj_2 = self.adj2(x,mode=mode)
        output = self.gcn(x_new, adj_1.detach(),  adj_2.detach())[:x.size()[0]]
        output = output+x
        return output
    
if __name__ == '__main__':
    x= torch.rand(10, 2048)
    adj_layer1 = ADJ_FirstLayer()
    adj_layer2 = ADJ_AttCenter()
    adj_1 = adj_layer1(x)
    x_new, adj_2 = adj_layer2(x)
    print(adj_1.shape)
    print(adj_2.shape)
    model = GCN_2ADJ(2048, 128, 1)
    output = model(x_new, adj_1, adj_2)
    print(output.shape)