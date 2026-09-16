#!/usr/bin/env python3
"""
Biometric identification using VA prediction features.

Performs subject/actor identification using features extracted from a VA backbone.
Tests whether identity information is implicitly captured during VA training.

Experimental setup:
  1. Use VA backbone features as input (frozen, not trained)
  2. Mean-pool the features per clip/video → identity embedding
  3. Subject classification with a simple classifier (Linear or MLP)
  4. Measure Accuracy, Top-5 Accuracy, EER, etc.

Datasets:
  - Aff-wild2: 68 val-only subjects, 1 video each → within-video temporal split
  - AFEW-VA: 67 actors (3+ clips) → cross-clip split (split_biometric.json)

Usage:
  # AFEW-VA biometric
  python eval_biometric_identification.py \
      --dataset afewva \
      --features_dir ./features/AFEWVA \
      --backbone ViViT \
      --split_file /path/to/split_biometric.json \
      --annotations_dir /path/to/preprocessed_VA_annotations/Train_Set

  # Aff-wild2 biometric
  python eval_biometric_identification.py \
      --dataset affwild2 \
      --features_dir ./features/CUSTOM_FINETUNED/val \
      --backbone ViViT_s4 \
      --annotations_dir /path/to/Aff-wild2/preprocessed_VA_annotations/Val_Set \
      --temporal_split_ratio 0.5
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# Datasets
# ============================================================================
class AFEWVABiometricDataset(Dataset):
    """AFEW-VA biometric dataset. Clip-level split based on split_biometric.json."""

    def __init__(self, features_dir, backbone, split_file, split='train'):
        """
        Args:
            split: 'train' or 'test'
        """
        with open(split_file) as f:
            split_data = json.load(f)

        self.samples = []
        self.label_encoder = LabelEncoder()
        actor_names = sorted(split_data['actors'].keys())
        self.label_encoder.fit(actor_names)
        self.num_classes = len(actor_names)

        backbone_dir = os.path.join(features_dir, backbone)

        for actor_name in actor_names:
            actor_info = split_data['actors'][actor_name]
            clips = actor_info['train_clips'] if split == 'train' else actor_info['test_clips']
            label = self.label_encoder.transform([actor_name])[0]

            for clip_id in clips:
                feat_path = os.path.join(backbone_dir, f"{clip_id}.pt")
                if os.path.exists(feat_path):
                    self.samples.append({
                        'clip_id': clip_id,
                        'actor': actor_name,
                        'label': label,
                        'feat_path': feat_path,
                    })

        print(f"[AFEWVABiometric-{split}] {len(self.samples)} clips, "
              f"{self.num_classes} actors")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        feat = torch.load(sample['feat_path'], weights_only=True)
        # Mean pooling over temporal dimension → (feat_dim,)
        feat_pooled = feat.mean(dim=0)
        return feat_pooled, sample['label'], sample['clip_id']


class Affwild2BiometricDataset(Dataset):
    """Aff-wild2 biometric dataset. Val-only 68 subjects, temporal split."""

    def __init__(self, features_dir, backbone, annotations_dir,
                 split='train', temporal_split_ratio=0.5, seed=42):
        """
        Within-video temporal split:
          train: the leading temporal_split_ratio portion
          test: the trailing (1 - temporal_split_ratio) portion
        The features are already a temporal sequence → split into leading/trailing chunks
        """
        self.samples = []
        backbone_dir = os.path.join(features_dir, backbone)
        rng = np.random.RandomState(seed)

        # Subject list from the val annotations
        annotation_files = sorted([f for f in os.listdir(annotations_dir)
                                    if f.endswith('.csv')])

        subject_ids = []
        for fname in annotation_files:
            video_id = os.path.splitext(fname)[0]
            feat_path = os.path.join(backbone_dir, f"{video_id}.pt")
            if os.path.exists(feat_path):
                subject_ids.append(video_id)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(subject_ids)
        self.num_classes = len(subject_ids)

        for video_id in subject_ids:
            feat_path = os.path.join(backbone_dir, f"{video_id}.pt")
            feat = torch.load(feat_path, weights_only=True)  # (T, feat_dim)
            label = self.label_encoder.transform([video_id])[0]

            T = feat.shape[0]
            split_point = max(1, int(T * temporal_split_ratio))

            if split == 'train':
                feat_split = feat[:split_point]
            else:
                feat_split = feat[split_point:]

            if feat_split.shape[0] == 0:
                continue

            # Chunk into segments for more training samples
            chunk_size = max(1, feat_split.shape[0] // 4)
            for start in range(0, feat_split.shape[0], chunk_size):
                end = min(start + chunk_size, feat_split.shape[0])
                chunk = feat_split[start:end]
                chunk_pooled = chunk.mean(dim=0)
                self.samples.append({
                    'video_id': video_id,
                    'label': label,
                    'feat': chunk_pooled,
                })

        print(f"[Affwild2Biometric-{split}] {len(self.samples)} chunks, "
              f"{self.num_classes} subjects")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['feat'], sample['label'], sample['video_id']


# ============================================================================
# Biometric Classifier
# ============================================================================
class BiometricClassifier(nn.Module):
    """Simple MLP classifier for biometric identification."""

    def __init__(self, in_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Training & Evaluation
# ============================================================================
def train_biometric(model, train_loader, val_loader, device, args):
    """Train biometric classifier."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for feats, labels, _ in train_loader:
            feats = feats.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

        scheduler.step()
        train_acc = correct / total

        # Validation
        val_acc, val_top5, val_results = evaluate_biometric(model, val_loader, device)

        if (epoch + 1) % 10 == 0 or val_acc > best_acc:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} (Top-5: {val_top5:.4f})")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc


@torch.no_grad()
def evaluate_biometric(model, loader, device):
    """Evaluate biometric classifier."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for feats, labels, _ in loader:
        feats = feats.to(device)
        if not isinstance(labels, torch.Tensor):
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            labels_t = labels

        logits = model(feats)
        probs = torch.softmax(logits, dim=-1).cpu()
        preds = logits.argmax(dim=-1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels_t.tolist())
        all_probs.append(probs)

    all_probs = torch.cat(all_probs, dim=0)

    # Top-1 accuracy
    acc = accuracy_score(all_labels, all_preds)

    # Top-5 accuracy
    all_labels_tensor = torch.tensor(all_labels)
    top5_preds = all_probs.topk(min(5, all_probs.shape[1]), dim=-1).indices
    top5_correct = sum(1 for i, label in enumerate(all_labels_tensor) if label in top5_preds[i])
    top5_acc = top5_correct / len(all_labels)

    return acc, top5_acc, {'preds': all_preds, 'labels': all_labels}


def collate_biometric(batch):
    feats = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    ids = [item[2] for item in batch]
    return feats, labels, ids


# ============================================================================
# Multi-seed Experiment
# ============================================================================
def run_multi_seed(args, device):
    """Repeat the experiment over several seeds and compute the mean/std."""
    results_all = []

    for seed in range(args.n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n--- Seed {seed} ---")

        # Dataset
        if args.dataset == 'afewva':
            train_dataset = AFEWVABiometricDataset(
                args.features_dir, args.backbone, args.split_file, split='train')
            test_dataset = AFEWVABiometricDataset(
                args.features_dir, args.backbone, args.split_file, split='test')
            num_classes = train_dataset.num_classes
            label_encoder = train_dataset.label_encoder
        elif args.dataset == 'affwild2':
            train_dataset = Affwild2BiometricDataset(
                args.features_dir, args.backbone, args.annotations_dir,
                split='train', temporal_split_ratio=args.temporal_split_ratio, seed=seed)
            test_dataset = Affwild2BiometricDataset(
                args.features_dir, args.backbone, args.annotations_dir,
                split='test', temporal_split_ratio=args.temporal_split_ratio, seed=seed)
            num_classes = train_dataset.num_classes
            label_encoder = train_dataset.label_encoder

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=collate_biometric)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_biometric)

        # Determine feature dim from first sample
        feat_dim = train_dataset[0][0].shape[0]

        # Model
        model = BiometricClassifier(
            in_dim=feat_dim, num_classes=num_classes,
            hidden_dim=args.hidden_dim, dropout=args.dropout
        ).to(device)

        # Train
        best_val_acc = train_biometric(model, train_loader, test_loader, device, args)

        # Final test
        test_acc, test_top5, test_results = evaluate_biometric(model, test_loader, device)

        print(f"Seed {seed}: Test Acc={test_acc:.4f}, Top-5={test_top5:.4f}")

        results_all.append({
            'seed': seed,
            'test_acc': test_acc,
            'test_top5': test_top5,
            'num_classes': num_classes,
            'n_train': len(train_dataset),
            'n_test': len(test_dataset),
        })

    return results_all, label_encoder


def main():
    parser = argparse.ArgumentParser(description="Biometric identification from VA features")
    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['afewva', 'affwild2'])
    parser.add_argument('--features_dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='ViViT')
    parser.add_argument('--annotations_dir', type=str, default=None,
                        help='Annotation dir (required for affwild2)')
    parser.add_argument('--split_file', type=str, default=None,
                        help='Biometric split JSON (required for afewva)')
    parser.add_argument('--temporal_split_ratio', type=float, default=0.5,
                        help='Aff-wild2 within-video temporal split ratio (default: 0.5)')
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    # Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Number of seeds to repeat the experiment over (default: 5)')
    # Output
    parser.add_argument('--save_dir', type=str, default='./biometric_results')
    parser.add_argument('--exp_name', type=str, default=None)

    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = f"{args.dataset}_{args.backbone}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Biometric Identification ===")
    print(f"  Dataset: {args.dataset}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Seeds: {args.n_seeds}")

    # Run
    results_all, label_encoder = run_multi_seed(args, device)

    # Aggregate
    accs = [r['test_acc'] for r in results_all]
    top5s = [r['test_top5'] for r in results_all]

    summary = {
        'exp_name': args.exp_name,
        'dataset': args.dataset,
        'backbone': args.backbone,
        'num_classes': results_all[0]['num_classes'],
        'n_seeds': args.n_seeds,
        'mean_acc': float(np.mean(accs)),
        'std_acc': float(np.std(accs)),
        'mean_top5': float(np.mean(top5s)),
        'std_top5': float(np.std(top5s)),
        'per_seed': results_all,
        'class_names': label_encoder.classes_.tolist(),
    }

    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Biometric Identification Results ({args.dataset}, {args.backbone})")
    print(f"  Classes: {results_all[0]['num_classes']}")
    print(f"  Top-1 Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Top-5 Accuracy: {np.mean(top5s):.4f} ± {np.std(top5s):.4f}")
    print(f"  Chance level: {1/results_all[0]['num_classes']:.4f}")
    print(f"{'='*60}")
    print(f"  Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
