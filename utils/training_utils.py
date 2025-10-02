"""
Training Utilities for Video Classification
===========================================
Functions for training loops, metrics, and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, top_k_accuracy_score
)


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch,
    config,
    scaler=None
):
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        config: Configuration dictionary
        scaler: GradScaler for mixed precision training
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    # Metrics
    losses = AverageMeter()
    accuracies = AverageMeter()
    batch_time = AverageMeter()
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    end_time = time.time()
    
    for batch_idx, (videos, labels) in enumerate(pbar):
        videos = videos.to(device)
        labels = labels.to(device)
        
        batch_size = videos.size(0)
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config['training'].get('gradient_clip', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['gradient_clip']
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            if config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['gradient_clip']
                )
            
            optimizer.step()
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        accuracy = (predicted == labels).float().mean().item()
        
        # Update metrics
        losses.update(loss.item(), batch_size)
        accuracies.update(accuracy, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return {
        'loss': losses.avg,
        'accuracy': accuracies.avg,
        'learning_rate': optimizer.param_groups[0]['lr']
    }


def validate(model, val_loader, criterion, device, config):
    """
    Validate model
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        config: Configuration dictionary
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    losses = AverageMeter()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            # Get predictions
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store results
            losses.update(loss.item(), videos.size(0))
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Top-5 accuracy
    top5_accuracy = top_k_accuracy_score(
        all_labels, all_probabilities, k=5, labels=range(all_probabilities.shape[1])
    )
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return {
        'loss': losses.avg,
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }


def test(model, test_loader, device, class_names: List[str] = None):
    """
    Test model and generate detailed report
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of class names
    
    Returns:
        Dictionary with test metrics and predictions
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for videos, labels in pbar:
            videos = videos.to(device)
            
            # Forward pass
            outputs = model(videos)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Top-5 accuracy
    top5_accuracy = top_k_accuracy_score(
        all_labels, all_probabilities, k=5, labels=range(all_probabilities.shape[1])
    )
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Overall metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    if class_names is not None:
        report = classification_report(
            all_labels, all_predictions,
            target_names=class_names,
            zero_division=0
        )
    else:
        report = classification_report(all_labels, all_predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1_score': f1_macro,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing
    """
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        n_classes = pred.size(-1)
        
        # One-hot encode targets
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        smoothed_labels = one_hot * confidence + (1 - one_hot) * self.smoothing / (n_classes - 1)
        
        # Calculate cross entropy
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(smoothed_labels * log_probs).sum(dim=-1).mean()
        
        return loss


class MixupCutmix:
    """
    Mixup and Cutmix data augmentation
    """
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=0.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def __call__(self, videos, labels):
        """
        Apply mixup or cutmix
        
        Args:
            videos: Tensor of shape (B, C, T, H, W)
            labels: Tensor of shape (B,)
        
        Returns:
            Mixed videos and labels
        """
        if np.random.rand() > self.prob:
            return videos, labels
        
        batch_size = videos.size(0)
        
        if self.cutmix_alpha > 0 and np.random.rand() < 0.5:
            # Apply Cutmix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            rand_index = torch.randperm(batch_size).to(videos.device)
            
            # Generate random box
            _, _, T, H, W = videos.shape
            cut_t = int(T * lam)
            cut_h = int(H * lam)
            cut_w = int(W * lam)
            
            t_start = np.random.randint(0, T - cut_t + 1)
            h_start = np.random.randint(0, H - cut_h + 1)
            w_start = np.random.randint(0, W - cut_w + 1)
            
            # Apply cutmix
            videos[:, :, t_start:t_start+cut_t, h_start:h_start+cut_h, w_start:w_start+cut_w] = \
                videos[rand_index, :, t_start:t_start+cut_t, h_start:h_start+cut_h, w_start:w_start+cut_w]
            
            # Mix labels
            lam = 1 - (cut_t * cut_h * cut_w) / (T * H * W)
            labels = (labels, labels[rand_index], lam)
        
        elif self.mixup_alpha > 0:
            # Apply Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            rand_index = torch.randperm(batch_size).to(videos.device)
            
            mixed_videos = lam * videos + (1 - lam) * videos[rand_index]
            labels = (labels, labels[rand_index], lam)
            
            return mixed_videos, labels
        
        return videos, labels


def mixup_criterion(criterion, pred, labels):
    """
    Calculate loss for mixed labels
    
    Args:
        criterion: Loss function
        pred: Model predictions
        labels: Tuple of (labels_a, labels_b, lam) or regular labels
    
    Returns:
        Loss value
    """
    if isinstance(labels, tuple):
        labels_a, labels_b, lam = labels
        return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)
    else:
        return criterion(pred, labels)

