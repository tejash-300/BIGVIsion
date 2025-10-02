"""
Visualization Utilities for Video Classification
================================================
Functions for visualizing results, metrics, and predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import torch
from typing import List, Optional, Dict
from pathlib import Path
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training history
    
    Args:
        history: Dictionary with training metrics over epochs
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy (handle both 'train_acc' and 'train_accuracy' keys)
    train_acc_key = 'train_acc' if 'train_acc' in history else 'train_accuracy'
    val_acc_key = 'val_acc' if 'val_acc' in history else 'val_accuracy'
    
    axes[0, 1].plot(history[train_acc_key], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history[val_acc_key], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot learning rate (handle both 'learning_rates' and 'learning_rate' keys)
    lr_key = 'learning_rates' if 'learning_rates' in history else 'learning_rate'
    if lr_key in history:
        axes[1, 0].plot(history[lr_key], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot top-5 accuracy if available
    if 'val_top5_accuracy' in history:
        axes[1, 1].plot(history['val_top5_accuracy'], linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Top-5 Accuracy')
        axes[1, 1].set_title('Validation Top-5 Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: tuple = (20, 16)
):
    """
    Plot confusion matrix
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize the matrix
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    
    # Determine font size based on number of classes
    num_classes = len(class_names)
    if num_classes > 50:
        font_size = 4
    elif num_classes > 30:
        font_size = 6
    else:
        font_size = 8
    
    sns.heatmap(
        conf_matrix,
        annot=True if num_classes <= 30 else False,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'},
        annot_kws={'size': font_size}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_per_class_metrics(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot per-class metrics
    
    Args:
        precision: Per-class precision scores
        recall: Per-class recall scores
        f1: Per-class F1 scores
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    # Create DataFrame
    df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # Sort by F1 score
    df = df.sort_values('F1-Score', ascending=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(class_names) * 0.3)))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    ax.barh(x - width, df['Precision'], width, label='Precision', alpha=0.8)
    ax.barh(x, df['Recall'], width, label='Recall', alpha=0.8)
    ax.barh(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(df['Class'], fontsize=8)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to {save_path}")
    
    plt.show()


def visualize_video_frames(
    video_tensor: torch.Tensor,
    label: int,
    prediction: int,
    class_names: List[str],
    num_frames: int = 8,
    save_path: Optional[str] = None
):
    """
    Visualize frames from a video with prediction
    
    Args:
        video_tensor: Video tensor of shape (C, T, H, W)
        label: True label
        prediction: Predicted label
        class_names: List of class names
        num_frames: Number of frames to display
        save_path: Optional path to save the plot
    """
    # Convert tensor to numpy and denormalize
    if isinstance(video_tensor, torch.Tensor):
        video = video_tensor.cpu().numpy()
    else:
        video = video_tensor
    
    # Handle different formats
    if video.shape[0] == 3:  # (C, T, H, W)
        video = video.transpose(1, 2, 3, 0)  # (T, H, W, C)
    
    # Normalize to [0, 1]
    if video.max() > 1.0:
        video = video / 255.0
    
    # Sample frames
    total_frames = video.shape[0]
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = video[indices]
    
    # Create plot
    fig, axes = plt.subplots(2, num_frames // 2, figsize=(16, 6))
    axes = axes.flatten()
    
    for idx, (frame, ax) in enumerate(zip(frames, axes)):
        ax.imshow(frame)
        ax.axis('off')
        ax.set_title(f'Frame {indices[idx]}', fontsize=10)
    
    # Add prediction info
    is_correct = (label == prediction)
    color = 'green' if is_correct else 'red'
    
    fig.suptitle(
        f'True: {class_names[label]} | Predicted: {class_names[prediction]} | {"✓" if is_correct else "✗"}',
        fontsize=14,
        fontweight='bold',
        color=color
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Video visualization saved to {save_path}")
    
    plt.show()


def plot_top_k_predictions(
    probabilities: np.ndarray,
    true_label: int,
    class_names: List[str],
    k: int = 5,
    save_path: Optional[str] = None
):
    """
    Plot top-k predictions for a sample
    
    Args:
        probabilities: Prediction probabilities for all classes
        true_label: True class label
        class_names: List of class names
        k: Number of top predictions to show
        save_path: Optional path to save the plot
    """
    # Get top-k predictions
    top_k_idx = np.argsort(probabilities)[-k:][::-1]
    top_k_probs = probabilities[top_k_idx]
    top_k_names = [class_names[idx] for idx in top_k_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if idx == true_label else 'steelblue' for idx in top_k_idx]
    
    bars = ax.barh(range(k), top_k_probs, color=colors, alpha=0.7)
    ax.set_yticks(range(k))
    ax.set_yticklabels(top_k_names)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title(f'Top-{k} Predictions (True label: {class_names[true_label]})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, top_k_probs)):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top-k predictions plot saved to {save_path}")
    
    plt.show()


def create_video_with_prediction(
    video_path: str,
    prediction: int,
    confidence: float,
    class_names: List[str],
    output_path: str,
    fps: int = 30
):
    """
    Create a video with prediction overlay
    
    Args:
        video_path: Path to input video
        prediction: Predicted class
        confidence: Prediction confidence
        class_names: List of class names
        output_path: Path to save output video
        fps: Frames per second for output video
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Prediction text
    pred_text = f'{class_names[prediction]}: {confidence:.2%}'
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add text overlay
        cv2.putText(
            frame, pred_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"Video with prediction saved to {output_path}")


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: Optional[str] = None
):
    """
    Plot class distribution
    
    Args:
        labels: Array of class labels
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save the plot
    """
    # Count classes
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Class': [class_names[i] for i in unique],
        'Count': counts
    })
    df = df.sort_values('Count', ascending=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(unique) * 0.3)))
    
    bars = ax.barh(df['Class'], df['Count'], color='steelblue', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', ha='left', va='center', fontsize=8)
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()

