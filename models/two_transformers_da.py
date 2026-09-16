"""
Fusion wrapper with Domain Adaptation support.

Two_transformers_DA: wraps the existing fusion model and adds the DA modules.
LightweightAttentionFusion: new lightweight fusion (~3M params).

Core multimodal transformers (mm_multi_transformers.py, mm_transformers.py) are not modified.
"""

from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .mm_multi_transformers import MultimodalTransformer_w_JR
    from .mm_multi_transformers import FeatureConcatFC
    from .mm_transformers import MultimodalTransformer_wo_JR
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The multimodal fusion transformer is not bundled with this repository. "
        "Obtain mm_transformers.py and mm_multi_transformers.py from the Joint "
        "Multimodal Transformer implementation cited in the paper and place them "
        "in models/. See README.md, \"Install\"."
    ) from exc


__all__ = ['Two_transformers_DA', 'LightweightAttentionFusion']


class LightweightAttentionFusion(nn.Module):
    """Lightweight fusion: 1-layer self-attention on V and A each → concat → FC → pred.
    ~3M params (vs JMT Full 16.5M)
    """
    def __init__(self, d_model=512, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()

        encoder_layer_v = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=False  # PyTorch 1.9 compatible: (seq, batch, feat)
        )
        self.vision_encoder = nn.TransformerEncoder(encoder_layer_v, num_layers=1)

        encoder_layer_a = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=False
        )
        self.audio_encoder = nn.TransformerEncoder(encoder_layer_a, num_layers=1)

    def forward(self, visual_features, audio_features):
        """
        Args:
            visual_features: (B, T, 512)
            audio_features: (B, T, 512)
        Returns:
            fused: (B, T, 1024) — concat of encoded V and A
        """
        # (B, T, D) → (T, B, D) for nn.TransformerEncoder
        v = visual_features.permute(1, 0, 2)
        a = audio_features.permute(1, 0, 2)

        v_enc = self.vision_encoder(v)  # (T, B, 512)
        a_enc = self.audio_encoder(a)   # (T, B, 512)

        # (T, B, D) → (B, T, D)
        v_enc = v_enc.permute(1, 0, 2)
        a_enc = a_enc.permute(1, 0, 2)

        # Concat → (B, T, 1024)
        fused = torch.cat((v_enc, a_enc), dim=2)
        return fused


class Two_transformers_DA(nn.Module):
    """Fusion wrapper with DA support.

    fusion_type options:
      - 'TRANSFORMER': JMT Full (MultimodalTransformer_w_JR) — existing
      - 'NONE': Vanilla (MultimodalTransformer_wo_JR) — existing
      - 'FC': FeatureConcatFC — existing
      - 'LIGHT_ATTN': LightweightAttentionFusion — new

    The DA modules are injected externally (losses/da_losses.py).
    forward returns the pre-fusion features so they can be used to compute the DA loss.
    """
    def __init__(self,
                 v_dropout: float,
                 a_dropout: float,
                 num_heads: int,
                 num_layers: int,
                 fusion_type: str = 'TRANSFORMER',
                 output_format: str = 'SELF_ATTEN',
                 vision_in_ft: int = 512):
        super().__init__()

        self.v_dropout = v_dropout
        self.a_dropout = a_dropout
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.fusion_type = fusion_type
        self.output_format = output_format
        self.vision_in_ft = vision_in_ft

        self.linear = None
        if vision_in_ft != 512:
            self.linear = nn.Linear(vision_in_ft, 512)

        # Select the fusion module
        if fusion_type == 'TRANSFORMER':
            assert output_format in ['FC', 'SELF_ATTEN'], output_format
            self.mm_transformer = MultimodalTransformer_w_JR(
                visual_dim=512, audio_dim=512,
                num_heads=num_heads, hidden_dim=512,
                num_layers=num_layers, output_format=output_format
            )
            dim = 1024 if output_format == 'FC' else 512

        elif fusion_type == 'NONE':
            assert output_format in ['FC'], output_format
            self.mm_transformer = MultimodalTransformer_wo_JR(
                visual_dim=512, audio_dim=512,
                num_heads=num_heads, hidden_dim=512,
                num_layers=num_layers, output_format=output_format
            )
            dim = 512

        elif fusion_type == 'FC':
            self.mm_transformer = FeatureConcatFC(512, 512)
            dim = 512

        elif fusion_type == 'LIGHT_ATTN':
            self.mm_transformer = LightweightAttentionFusion(
                d_model=512, nhead=num_heads,
                dim_feedforward=512, dropout=v_dropout
            )
            dim = 1024  # concat of V(512) + A(512)

        else:
            raise NotImplementedError(f"Unknown fusion_type: {fusion_type}")

        # Regressor heads
        self.vregressor = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(v_dropout),
            nn.Linear(128, 1)
        )
        self.aregressor = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(a_dropout),
            nn.Linear(128, 1)
        )

    def forward(self, video, audio):
        """
        Args:
            video: (B, T, feat) — L2 normalized, after DA
            audio: (B, T, 512)  — L2 normalized, after DA

        Returns:
            dict with keys:
                'pred_v': (B, T) — valence prediction
                'pred_a': (B, T) — arousal prediction
                'video_pre_fusion': (B, T, 512) — for DA losses
                'audio_pre_fusion': (B, T, 512) — for DA losses
        """
        # L2 normalize
        video_norm = F.normalize(video, dim=-1)
        audio_norm = F.normalize(audio, dim=-1)

        if self.linear is not None:
            video_norm = self.linear(video_norm)

        # Keep the pre-fusion features (for the DA loss)
        video_pre_fusion = video_norm
        audio_pre_fusion = audio_norm

        # Fusion
        fused = self.mm_transformer(video_norm, audio_norm)

        # Regression
        pred_v = self.vregressor(fused).squeeze(2)  # (B, T)
        pred_a = self.aregressor(fused).squeeze(2)  # (B, T)

        return {
            'pred_v': pred_v,
            'pred_a': pred_a,
            'video_pre_fusion': video_pre_fusion,
            'audio_pre_fusion': audio_pre_fusion,
        }
