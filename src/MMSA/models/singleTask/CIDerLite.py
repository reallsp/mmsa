"""
CIDerLite (Regression)

目标：把 CIDer 的“轻量融合思想”接入 MMSA 框架，并对齐 TFN 的训练/测试流程与指标。

输入：
- text:  (B, L_t, 768)  来自 copa_*_converted.pkl 的 `text`
- audio: (B, L_a, 13)
- vision:(B, L_v, 768)
- extras: dict，可包含 ir_feature/bio/eye/eeg/eda 等 (B, L, D)

输出：
- dict: { 'Feature_t', 'Feature_a', 'Feature_v', 'Feature_f', 'M' }
其中 M 为回归输出，范围约束到 [-1, 1]（与 copa_1231 regression_labels 对齐）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIDerLite(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # base dims from dataloader / config
        self.text_in, self.audio_in, self.vision_in = args.feature_dims

        # hyper-params
        self.embed_dim = int(getattr(args, "embed_dim", 128))
        self.num_heads = int(getattr(args, "num_heads", 4))
        self.layers = int(getattr(args, "layers", 2))
        self.dropout = float(getattr(args, "dropout", 0.1))
        self.rnn_hidden = int(getattr(args, "rnn_hidden", 128))
        self.extra_hidden = int(getattr(args, "extra_hidden", self.embed_dim))

        # ===== encoders =====
        # text sequence encoder
        self.text_rnn = nn.GRU(
            input_size=self.text_in,
            hidden_size=self.rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.rnn_hidden * 2, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # audio / vision: mean-pool over time then project
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_in, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_in, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # ===== extra modality encoders (mean-pool + linear) =====
        # key -> input dim
        extra_in_dims = {
            "ir_feature": self.vision_in,  # (B, 197, 768)
            "bio": 3,
            "eye": 2,
            "eeg": 8,
            "eda": 7,
        }
        self.extra_mlps = nn.ModuleDict(
            {k: nn.Linear(d, self.extra_hidden) for k, d in extra_in_dims.items()}
        )
        self.extra_proj = nn.Sequential(
            nn.Linear(len(self.extra_mlps) * self.extra_hidden, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # ===== fusion transformer over modality tokens =====
        # tokens: [CLS, T, A, V, EX]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)

        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, 1),
        )

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _masked_mean(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D), lengths: (B,)
        """
        if x.dim() != 3:
            raise ValueError(f"masked_mean expects 3D tensor, got dim={x.dim()}")
        lengths = lengths.to(x.device).long().clamp(min=1)
        bsz, max_len, _ = x.shape
        mask = torch.arange(max_len, device=x.device).unsqueeze(0).expand(bsz, -1) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _mean_pool(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, D) or (B, D)
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            if lengths is None:
                return x.mean(dim=1)
            return self._masked_mean(x, lengths)
        raise ValueError(f"Unexpected tensor dim for pooling: {x.dim()}")

    def _encode_text(self, text: torch.Tensor) -> torch.Tensor:
        # text: (B, L, D)
        out, h = self.text_rnn(text)  # h: (2, B, H)
        h = torch.cat([h[0], h[1]], dim=-1)  # (B, 2H)
        return self.text_proj(h)  # (B, E)

    def _encode_extras(
        self,
        extras: dict | None,
        extras_lengths: dict | None,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        # returns (B, E)
        if not extras:
            return torch.zeros((batch_size, self.embed_dim), device=device)
        encoded = []
        for k, mlp in self.extra_mlps.items():
            if k not in extras:
                encoded.append(torch.zeros((batch_size, self.extra_hidden), device=device))
                continue
            x = extras[k]
            x_len = None
            if extras_lengths and (f"{k}_lengths" in extras_lengths):
                x_len = extras_lengths[f"{k}_lengths"]
            x = self._mean_pool(x, lengths=x_len)
            encoded.append(F.relu(mlp(x)))
        concat = torch.cat(encoded, dim=-1)
        return self.extra_proj(concat)

    def forward(self, text_x, audio_x, video_x, extras=None, lengths=None, *args, **kwargs):
        """
        Args:
            text_x:  (B, L_t, D_t)
            audio_x: (B, L_a, D_a)
            video_x: (B, L_v, D_v)
            extras: dict[str, Tensor]
            lengths: dict，可包含 audio_lengths/vision_lengths 以及 bio_lengths 等
        """
        lengths = lengths or {}

        # encode base modalities
        t = self._encode_text(text_x)
        a = self.audio_proj(self._mean_pool(audio_x, lengths=lengths.get("audio_lengths")))
        v = self.vision_proj(self._mean_pool(video_x, lengths=lengths.get("vision_lengths")))

        batch_size = t.shape[0]
        device = t.device
        e = self._encode_extras(
            extras,
            extras_lengths=lengths,
            device=device,
            batch_size=batch_size,
        )

        # fuse with transformer tokens
        cls = self.cls_token.expand(batch_size, 1, -1)
        tokens = torch.stack([t, a, v, e], dim=1)  # (B, 4, E)
        x = torch.cat([cls, tokens], dim=1)  # (B, 5, E)
        x = self.fusion(x)  # (B, 5, E)
        fused = x[:, 0, :]  # CLS

        out = self.head(fused)  # (B, 1)
        # NOTE:
        # 使用 tanh 会在全量数据上很容易饱和到 +/-1，造成常数预测（梯度几乎为0），从而测试指标长期不变。
        # 因此默认不做输出激活；在评估阶段我们会对用于 COPA 的预测做 clip 到 [-1, 1]。

        return {
            "Feature_t": t,
            "Feature_a": a,
            "Feature_v": v,
            "Feature_f": fused,
            "M": out,
        }

