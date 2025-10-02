"""
Data Utilities for Video Classification
========================================
Functions for loading, processing, and augmenting video data.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import decord
from decord import VideoReader, cpu
from typing import List, Tuple, Optional, Dict
import albumentations as A
from pathlib import Path
import json


class VideoDataset(Dataset):
    """
    Custom Dataset for Video Classification
    
    Supports multiple video formats and efficient frame sampling.
    """
    
    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        sampling_strategy: str = 'uniform',
        transform=None,
        augment: bool = False
    ):
        """
        Args:
            video_paths: List of paths to video files
            labels: List of class labels
            num_frames: Number of frames to sample from each video
            frame_size: Target frame size (H, W)
            sampling_strategy: 'uniform', 'random', or 'dense'
            transform: Optional transform to apply
            augment: Whether to apply data augmentation
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sampling_strategy = sampling_strategy
        self.transform = transform
        self.augment = augment
        
        # Set decord to use CPU for video decoding
        decord.bridge.set_bridge('torch')
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """Load and process a video"""
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load video frames
            frames = self._load_video(video_path)
            
            # Apply transformations
            if self.transform:
                frames = self.transform(frames)
            
            # Apply augmentation if enabled
            if self.augment:
                frames = self._augment_frames(frames)
            
            # Convert to tensor if not already
            if not isinstance(frames, torch.Tensor):
                frames = torch.from_numpy(frames).float()
            
            # Normalize to [0, 1] if needed
            if frames.max() > 1.0:
                frames = frames / 255.0
            
            # Rearrange dimensions to (C, T, H, W) if needed
            if frames.ndim == 4 and frames.shape[0] != 3:  # (T, H, W, C)
                frames = frames.permute(3, 0, 1, 2)
            
            return frames, label
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return zeros as fallback
            frames = torch.zeros(3, self.num_frames, *self.frame_size)
            return frames, label
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """
        Load video and sample frames
        
        Returns:
            frames: numpy array of shape (T, H, W, C)
        """
        try:
            # Use decord for efficient video loading
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # Sample frame indices based on strategy
            if self.sampling_strategy == 'uniform':
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            elif self.sampling_strategy == 'random':
                indices = np.sort(np.random.choice(total_frames, self.num_frames, replace=False))
            else:  # dense
                start_idx = np.random.randint(0, max(1, total_frames - self.num_frames))
                indices = np.arange(start_idx, start_idx + self.num_frames)
            
            # Load frames
            frames = vr.get_batch(indices).asnumpy()
            
            # Resize frames
            frames = np.stack([
                cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
                for frame in frames
            ])
            
            return frames
            
        except Exception as e:
            # Fallback to OpenCV
            return self._load_video_opencv(video_path)
    
    def _load_video_opencv(self, video_path: str) -> np.ndarray:
        """Fallback video loading using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate indices
        if self.sampling_strategy == 'uniform':
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.random.choice(total_frames, self.num_frames, replace=False)
            indices = np.sort(indices)
        
        # Read frames
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
        
        cap.release()
        
        # Handle case where we couldn't read enough frames
        while len(frames) < self.num_frames:
            frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        return np.array(frames)
    
    def _augment_frames(self, frames: np.ndarray) -> np.ndarray:
        """Apply data augmentation to video frames"""
        # Define augmentation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            A.Rotate(limit=15, p=0.3),
        ])
        
        # Apply to each frame
        augmented_frames = []
        for frame in frames:
            augmented = transform(image=frame)['image']
            augmented_frames.append(augmented)
        
        return np.array(augmented_frames)


def create_data_loaders(
    config: Dict,
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: Optional[List[str]] = None,
    test_labels: Optional[List[int]] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test data loaders
    
    Args:
        config: Configuration dictionary
        train_paths, train_labels: Training data
        val_paths, val_labels: Validation data
        test_paths, test_labels: Optional test data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = VideoDataset(
        train_paths,
        train_labels,
        num_frames=config['video']['num_frames'],
        frame_size=tuple(config['video']['frame_size']),
        sampling_strategy=config['video']['sampling_strategy'],
        augment=True
    )
    
    val_dataset = VideoDataset(
        val_paths,
        val_labels,
        num_frames=config['video']['num_frames'],
        frame_size=tuple(config['video']['frame_size']),
        sampling_strategy='uniform',
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    test_loader = None
    if test_paths is not None and test_labels is not None:
        test_dataset = VideoDataset(
            test_paths,
            test_labels,
            num_frames=config['video']['num_frames'],
            frame_size=tuple(config['video']['frame_size']),
            sampling_strategy='uniform',
            augment=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
    
    return train_loader, val_loader, test_loader


def prepare_ucf101_dataset(data_root: str, splits_dir: str):
    """
    Prepare UCF101 dataset structure
    
    Args:
        data_root: Root directory containing UCF101 videos
        splits_dir: Directory to save train/val/test splits
    """
    from sklearn.model_selection import train_test_split
    
    # Get all video files
    video_files = []
    labels = []
    class_names = []
    
    data_path = Path(data_root)
    
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            class_idx = len(class_names)
            class_names.append(class_name)
            
            for video_file in class_dir.glob('*.avi'):
                video_files.append(str(video_file))
                labels.append(class_idx)
    
    # Create train/val/test splits
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        video_files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Save splits
    os.makedirs(splits_dir, exist_ok=True)
    
    splits = {
        'train': {'files': train_files, 'labels': train_labels},
        'val': {'files': val_files, 'labels': val_labels},
        'test': {'files': test_files, 'labels': test_labels},
        'class_names': class_names
    }
    
    for split_name in ['train', 'val', 'test']:
        split_data = {
            'files': splits[split_name]['files'],
            'labels': splits[split_name]['labels']
        }
        with open(os.path.join(splits_dir, f'{split_name}.json'), 'w') as f:
            json.dump(split_data, f, indent=2)
    
    # Save class names
    with open(os.path.join(splits_dir, 'class_names.json'), 'w') as f:
        json.dump({'class_names': class_names}, f, indent=2)
    
    print(f"Dataset prepared:")
    print(f"  Train: {len(train_files)} videos")
    print(f"  Val: {len(val_files)} videos")
    print(f"  Test: {len(test_files)} videos")
    print(f"  Classes: {len(class_names)}")
    
    return splits


def load_splits(splits_dir: str):
    """Load train/val/test splits"""
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        split_file = os.path.join(splits_dir, f'{split_name}.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits[split_name] = json.load(f)
    
    # Load class names
    with open(os.path.join(splits_dir, 'class_names.json'), 'r') as f:
        class_data = json.load(f)
        splits['class_names'] = class_data['class_names']
    
    return splits

