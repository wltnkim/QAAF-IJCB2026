"""
Domain Adaptation losses for JMT fusion model.

Includes:
- ModalityDropout: zero out one modality during training (fixed probability)
- AdaptiveModalityDropout: quality-based adaptive dropout (learned probability per sample)
- QualityAwareGating: soft modality scaling based on estimated quality (learned MLP)
- QMFGating: energy-based modality scaling (Zhang et al., ICML 2023, no learnable params)
- InfoNCEContrastiveLoss: align V/A feature spaces
- GradientReversalFunction + ModalityDiscriminator + AdversarialModalityLoss: modality-invariant features
- UnimodalPredictionHeads: V-only/A-only prediction + consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class QualityAwareGating(nn.Module):
    """Quality-aware soft gating: scales each modality's contribution by a learned quality score.

    Per-sample quality estimation → soft scaling.
    Gradients flow directly into the quality estimator through the soft gating → stable training.

    Active in both training and inference (soft scaling is always applied).
    ~33K params per modality.
    """
    def __init__(self, input_dim=512, hidden_dim=64):
        super().__init__()
        self.video_gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.audio_gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, video, audio):
        """
        Args:
            video: (B, T, D)
            audio: (B, T, D)
        Returns:
            video_gated: (B, T, D) — soft-scaled
            audio_gated: (B, T, D) — soft-scaled
            gate_info: dict with 'v_gate' (B, 1), 'a_gate' (B, 1)
        """
        # Mean pool over time → (B, D) → quality score (B, 1)
        v_gate = self.video_gate_net(video.mean(dim=1))  # (B, 1)
        a_gate = self.audio_gate_net(audio.mean(dim=1))  # (B, 1)

        # Soft scaling: feature × gate
        video_gated = video * v_gate.unsqueeze(1)   # (B, T, D)
        audio_gated = audio * a_gate.unsqueeze(1)   # (B, T, D)

        return video_gated, audio_gated, {
            'v_gate': v_gate,
            'a_gate': a_gate,
        }


class QMFGating(nn.Module):
    """QMF (Zhang et al., ICML 2023) baseline: energy-based modality gating.

    Applies the core idea of Quality-aware Multimodal Fusion to a regression task.
    The original QMF uses the energy score of classification logits, whereas
    for regression the L2 norm of the features is used as the energy proxy.

    Energy proxy = L2 norm of mean-pooled features (higher = more confident).
    Fusion weights = softmax(energy) across modalities.

    **No learnable parameters** (the key difference from QAG).
    Active in both training and inference.
    """
    def __init__(self):
        super().__init__()
        # No learnable parameters

    def forward(self, video, audio):
        """
        Args:
            video: (B, T, D)
            audio: (B, T, D)
        Returns:
            video_gated: (B, T, D) — soft-scaled
            audio_gated: (B, T, D) — soft-scaled
            gate_info: dict with 'v_gate' (B, 1), 'a_gate' (B, 1)
        """
        # Mean pool over time → (B, D)
        v_pooled = video.mean(dim=1)  # (B, D)
        a_pooled = audio.mean(dim=1)  # (B, D)

        # Energy proxy: L2 norm of mean-pooled features (B,)
        v_energy = torch.norm(v_pooled, p=2, dim=-1, keepdim=True)  # (B, 1)
        a_energy = torch.norm(a_pooled, p=2, dim=-1, keepdim=True)  # (B, 1)

        # Softmax across modalities → fusion weights
        energies = torch.cat([v_energy, a_energy], dim=-1)  # (B, 2)
        weights = F.softmax(energies, dim=-1)  # (B, 2)

        v_gate = weights[:, 0:1]  # (B, 1)
        a_gate = weights[:, 1:2]  # (B, 1)

        # Soft scaling: feature × gate (scale by 2 to preserve magnitude,
        # since softmax weights sum to 1 and equal weights would give 0.5 each)
        video_gated = video * (v_gate * 2).unsqueeze(1)   # (B, T, D)
        audio_gated = audio * (a_gate * 2).unsqueeze(1)   # (B, T, D)

        return video_gated, audio_gated, {
            'v_gate': v_gate,
            'a_gate': a_gate,
        }


class FixedGating(nn.Module):
    """Fixed constant gating baseline: gate=0.5 for both modalities (no learning).

    Control experiment testing whether QAG's performance is reachable without any learning.
    Same interface as QAG (forward → gated features + gate_info).
    No learnable parameters.
    """
    def __init__(self, gate_value=0.5):
        super().__init__()
        self.gate_value = gate_value

    def forward(self, video, audio):
        B = video.shape[0]
        v_gate = torch.full((B, 1), self.gate_value, device=video.device)
        a_gate = torch.full((B, 1), self.gate_value, device=audio.device)

        video_gated = video * self.gate_value
        audio_gated = audio * self.gate_value

        return video_gated, audio_gated, {
            'v_gate': v_gate,
            'a_gate': a_gate,
        }


class AdaptiveModalityDropout(nn.Module):
    """Quality-based adaptive modality dropout.

    Per-sample quality → dropout probability:
      high quality → low dropout (trusted, kept)
      low quality → high dropout (distrusted, removed)

    p_drop = p_max - (p_max - p_min) * quality
      quality=1 → p_drop=p_min (minimum dropout)
      quality=0 → p_drop=p_max (maximum dropout)

    With learnable_pmax=True, p_max is learned as a per-modality parameter:
      p_max = sigmoid(logit) * (range_max - p_min) + p_min
      the optimal dropout strength differs per backbone → decided automatically, no manual tuning.

    Takes an external quality score (QualityAwareGating) or estimates one internally.
    Training only (pass-through at eval time).
    ~33K params (when using the internal quality estimator).
    """
    def __init__(self, input_dim=512, hidden_dim=64, p_min=0.05, p_max=0.4,
                 learnable_pmax=False, pmax_range_max=0.7):
        super().__init__()
        self.p_min = p_min
        self.learnable_pmax = learnable_pmax

        if learnable_pmax:
            # p_max = sigmoid(logit) * (range_max - p_min) + p_min
            # Initialize so that sigmoid(logit) maps to the given p_max default
            range_size = pmax_range_max - p_min
            init_sigmoid = (p_max - p_min) / range_size  # e.g., (0.4-0.05)/(0.7-0.05) ≈ 0.538
            init_logit = torch.logit(torch.tensor(init_sigmoid).clamp(0.01, 0.99))
            self._pmax_logit_v = nn.Parameter(init_logit.clone())
            self._pmax_logit_a = nn.Parameter(init_logit.clone())
            self._pmax_range_min = p_min
            self._pmax_range_max = pmax_range_max
        else:
            self.p_max = p_max

        # Internal quality estimator (used when no external quality is provided)
        self.video_quality_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.audio_quality_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def get_p_max(self):
        """Return current (v_pmax, a_pmax). Passed through a sigmoid when learnable."""
        if self.learnable_pmax:
            range_size = self._pmax_range_max - self._pmax_range_min
            v_pmax = torch.sigmoid(self._pmax_logit_v) * range_size + self._pmax_range_min
            a_pmax = torch.sigmoid(self._pmax_logit_a) * range_size + self._pmax_range_min
            return v_pmax, a_pmax
        else:
            return self.p_max, self.p_max

    def forward(self, video, audio, v_quality=None, a_quality=None):
        """
        Args:
            video: (B, T, D)
            audio: (B, T, D)
            v_quality: (B, 1) optional — external quality score (e.g., from QualityAwareGating)
            a_quality: (B, 1) optional
        Returns:
            video, audio (possibly zeroed), quality_info dict
        """
        if not self.training:
            return video, audio, {}

        B = video.shape[0]

        # Quality estimation (estimated internally when not provided externally)
        if v_quality is None:
            v_quality = self.video_quality_net(video.mean(dim=1))  # (B, 1)
        if a_quality is None:
            a_quality = self.audio_quality_net(audio.mean(dim=1))  # (B, 1)

        # Quality → dropout probability (supports a learnable p_max)
        v_pmax, a_pmax = self.get_p_max()
        v_p = v_pmax - (v_pmax - self.p_min) * v_quality  # (B, 1)
        a_p = a_pmax - (a_pmax - self.p_min) * a_quality  # (B, 1)

        # Per-sample binary dropout
        v_rand = torch.rand(B, 1, device=video.device)
        a_rand = torch.rand(B, 1, device=audio.device)

        v_keep = (v_rand >= v_p).float().unsqueeze(-1)  # (B, 1, 1)
        a_keep = (a_rand >= a_p).float().unsqueeze(-1)  # (B, 1, 1)

        # Prevent both from being dropped → keep the higher-quality one
        both_dropped = (v_keep.view(B) == 0) & (a_keep.view(B) == 0)
        if both_dropped.any():
            keep_v = (v_quality.squeeze(-1) >= a_quality.squeeze(-1))[both_dropped]
            v_keep[both_dropped] = keep_v.float().view(-1, 1, 1)
            a_keep[both_dropped] = (~keep_v).float().view(-1, 1, 1)

        video = video * v_keep
        audio = audio * a_keep

        info = {
            'v_quality': v_quality.detach(),
            'a_quality': a_quality.detach(),
            'v_drop_prob': v_p.detach(),
            'a_drop_prob': a_p.detach(),
        }
        if self.learnable_pmax:
            info['v_pmax'] = v_pmax.detach()
            info['a_pmax'] = a_pmax.detach()
        return video, audio, info


class ModalityDropout(nn.Module):
    """Stochastically zeroes one modality during training.
    Improves robustness to a missing modality.
    Extra params: 0
    """
    def __init__(self, p_drop=0.15):
        super().__init__()
        self.p_drop = p_drop

    def forward(self, video, audio):
        """
        Args:
            video: (B, T, 512)
            audio: (B, T, 512)
        Returns:
            video, audio (possibly zeroed)
        """
        if not self.training or self.p_drop <= 0:
            return video, audio

        # Decide the drop independently for each sample
        B = video.shape[0]
        rand = torch.rand(B, device=video.device)

        # Drop video with probability p_drop/2, audio with probability p_drop/2
        video_mask = (rand >= self.p_drop / 2).float().view(B, 1, 1)
        audio_mask = ((rand < self.p_drop / 2) | (rand >= self.p_drop)).float().view(B, 1, 1)

        video = video * video_mask
        audio = audio * audio_mask

        return video, audio


class InfoNCEContrastiveLoss(nn.Module):
    """Symmetric InfoNCE for aligning the V/A feature spaces.
    Mean pool over time → (B, 512) → similarity matrix → symmetric CE.
    Extra params: 0
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, video, audio):
        """
        Args:
            video: (B, T, 512)
            audio: (B, T, 512)
        Returns:
            loss: scalar
        """
        B = video.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=video.device, requires_grad=True)

        # Mean pool over time
        v_pooled = F.normalize(video.mean(dim=1), dim=-1)  # (B, 512)
        a_pooled = F.normalize(audio.mean(dim=1), dim=-1)  # (B, 512)

        # Cosine similarity matrix
        logits = torch.mm(v_pooled, a_pooled.t()) / self.temperature  # (B, B)

        labels = torch.arange(B, device=video.device)

        # Symmetric InfoNCE
        loss_v2a = F.cross_entropy(logits, labels)
        loss_a2v = F.cross_entropy(logits.t(), labels)

        return (loss_v2a + loss_a2v) / 2


class GradientReversalFunction(Function):
    """Gradient Reversal Layer: forward pass-through, backward -lambda * grad."""
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class ModalityDiscriminator(nn.Module):
    """Small classifier that discriminates between the modalities.
    Linear(512, 64) → ReLU → Linear(64, 1)
    ~33K params
    """
    def __init__(self, input_dim=512, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (N, 512) — flattened V or A features
        Returns:
            logits: (N, 1)
        """
        return self.net(x)


class AdversarialModalityLoss(nn.Module):
    """Combination of GRL and ModalityDiscriminator.
    Trains the V/A features to be modality-invariant.
    ~33K params
    """
    def __init__(self, input_dim=512, hidden_dim=64):
        super().__init__()
        self.discriminator = ModalityDiscriminator(input_dim, hidden_dim)

    def forward(self, video, audio, lambda_val=1.0):
        """
        Args:
            video: (B, T, 512)
            audio: (B, T, 512)
            lambda_val: GRL lambda (can be scheduled)
        Returns:
            loss: scalar (BCE loss)
        """
        # Mean pool over time
        v_pooled = video.mean(dim=1)  # (B, 512)
        a_pooled = audio.mean(dim=1)  # (B, 512)

        # Apply GRL
        v_reversed = GradientReversalFunction.apply(v_pooled, lambda_val)
        a_reversed = GradientReversalFunction.apply(a_pooled, lambda_val)

        # Discriminator predictions
        v_logits = self.discriminator(v_reversed)  # (B, 1)
        a_logits = self.discriminator(a_reversed)  # (B, 1)

        # Labels: video=1, audio=0
        v_labels = torch.ones_like(v_logits)
        a_labels = torch.zeros_like(a_logits)

        loss = F.binary_cross_entropy_with_logits(v_logits, v_labels) + \
               F.binary_cross_entropy_with_logits(a_logits, a_labels)

        return loss / 2


class UnimodalPredictionHeads(nn.Module):
    """V-only and A-only prediction heads + consistency loss.
    v_head: Linear(512, 64) → ReLU → Linear(64, 2)
    a_head: same structure
    ~33K params total
    """
    def __init__(self, input_dim=512, hidden_dim=64):
        super().__init__()
        self.v_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )
        self.a_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, video, audio):
        """
        Args:
            video: (B, T, 512)
            audio: (B, T, 512)
        Returns:
            v_pred: (B, T, 2) — V-only [valence, arousal]
            a_pred: (B, T, 2) — A-only [valence, arousal]
        """
        v_pred = self.v_head(video)
        a_pred = self.a_head(audio)
        return v_pred, a_pred

    def consistency_loss(self, v_pred, a_pred):
        """V-only pred ≈ A-only pred (MSE).
        Args:
            v_pred: (B, T, 2)
            a_pred: (B, T, 2)
        Returns:
            loss: scalar
        """
        return F.mse_loss(v_pred, a_pred)


class CrossModalReconstruction(nn.Module):
    """Cross-Modal Reconstruction (CMR).
    Predicts the audio feature from the video feature, and vice versa.
    Learning the cross-modal correspondence strengthens the feature representation.
    ~525K params total (2 decoders).
    """
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.v2a_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )
        self.a2v_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, video, audio):
        """
        Args:
            video: (B, T, D) — pre-fusion video features
            audio: (B, T, D) — pre-fusion audio features
        Returns:
            loss: scalar — bidirectional reconstruction MSE
        """
        audio_pred = self.v2a_decoder(video)
        video_pred = self.a2v_decoder(audio)
        loss_v2a = F.mse_loss(audio_pred, audio.detach())
        loss_a2v = F.mse_loss(video_pred, video.detach())
        return (loss_v2a + loss_a2v) / 2


class CrossModalDistillation(nn.Module):
    """Cross-Modal Knowledge Distillation (CMKD).
    The multimodal prediction (teacher) teaches the unimodal prediction (student).
    Trains each modality to make a good prediction on its own.
    Requires UnimodalPredictionHeads.
    No additional params (uses existing unimodal heads).
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, uni_v_pred, uni_a_pred, multimodal_pred_v, multimodal_pred_a):
        """
        Args:
            uni_v_pred: (B, T, 2) — V-only [valence, arousal]
            uni_a_pred: (B, T, 2) — A-only [valence, arousal]
            multimodal_pred_v: (B, T) — multimodal valence prediction
            multimodal_pred_a: (B, T) — multimodal arousal prediction
        Returns:
            loss: scalar — MSE distillation loss
        """
        # Stack multimodal predictions to (B, T, 2)
        mm_pred = torch.stack([multimodal_pred_v, multimodal_pred_a], dim=-1).detach()
        # Distill: unimodal should match multimodal
        loss_v = F.mse_loss(uni_v_pred, mm_pred)
        loss_a = F.mse_loss(uni_a_pred, mm_pred)
        return (loss_v + loss_a) / 2


def grl_lambda_schedule(epoch, max_epochs=50):
    """GRL lambda scheduling: λ = 2/(1+exp(-10p))-1, p=epoch/max_epochs.
    Starts near 0, gradually increases to ~1.
    """
    import math
    p = epoch / max_epochs
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
