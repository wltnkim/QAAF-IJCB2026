# datasets/dataset_features.py

import os
import torch
import torch.utils.data as data
import pandas as pd
from os.path import join

class FeatureDataset(data.Dataset):
    def __init__(self, features_dir, annotation_dir, vision_backbones, audio_backbones):
        self.features_dir = features_dir
        self.vision_backbones = vision_backbones
        self.audio_backbones = audio_backbones
        
        self.samples = []
        annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.csv')])

        for fname in annotation_files:
            video_id = os.path.splitext(fname)[0]
            # Check that all feature files exist
            feature_paths = self._get_feature_paths(video_id)
            if all(os.path.exists(p) for p in feature_paths.values()):
                # Read the labels from the annotation file
                df = pd.read_csv(join(annotation_dir, fname))
                labels_V = torch.tensor(df['V'].values, dtype=torch.float32)
                labels_A = torch.tensor(df['A'].values, dtype=torch.float32)
                self.samples.append({'video_id': video_id, 'labels': (labels_V, labels_A)})

    def _get_feature_paths(self, video_id):
        paths = {}
        for backbone in self.vision_backbones:
            paths[backbone] = join(self.features_dir, backbone, f"{video_id}.pt")
        for backbone in self.audio_backbones:
            paths[backbone] = join(self.features_dir, backbone, f"{video_id}.pt")
        return paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        video_id = sample['video_id']
        labels = sample['labels']
        
        feature_paths = self._get_feature_paths(video_id)
        
        features = {}
        for backbone, path in feature_paths.items():
            features[backbone] = torch.load(path)
            
        vision_features_list = [features[backbone] for backbone in self.vision_backbones]
        # Concatenate the tensors along the last dimension (feature dimension).
        vision_feature = torch.cat(vision_features_list, dim=-1)

        # Only one audio feature is used for now.
        audio_feature = features[self.audio_backbones[0]]
        # [end of modified section]
        
        return vision_feature, audio_feature, labels