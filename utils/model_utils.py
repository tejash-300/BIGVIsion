"""
Model Utilities for Video Classification
=========================================
Functions for creating, loading, and managing models.
"""

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, TimesformerForVideoClassification
from transformers import VideoMAEConfig, TimesformerConfig
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Optional
import timm


class VideoMAEClassifier(nn.Module):
    """VideoMAE model for video classification"""
    
    def __init__(self, num_classes: int, config: Dict, pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            # Load pretrained VideoMAE
            self.model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Create from config
            model_config = VideoMAEConfig(
                num_labels=num_classes,
                **config.get('videomae', {})
            )
            self.model = VideoMAEForVideoClassification(model_config)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, T, H, W)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        # VideoMAE expects (B, T, C, H, W), so permute from (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
        outputs = self.model(x)
        return outputs.logits


class TimeSformerClassifier(nn.Module):
    """TimeSformer model for video classification"""
    
    def __init__(self, num_classes: int, config: Dict, pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            # Load pretrained TimeSformer
            self.model = TimesformerForVideoClassification.from_pretrained(
                "facebook/timesformer-base-finetuned-k400",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Create from config
            model_config = TimesformerConfig(
                num_labels=num_classes,
                **config.get('timesformer', {})
            )
            self.model = TimesformerForVideoClassification(model_config)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, T, H, W)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        outputs = self.model(x)
        return outputs.logits


class I3DClassifier(nn.Module):
    """I3D model for video classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        
        # Use a 3D ResNet as backbone (similar to I3D)
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Convert 2D convolutions to 3D
        self._convert_to_3d()
        
        # Classification head
        self.fc = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def _convert_to_3d(self):
        """Convert 2D CNN to 3D (I3D-style inflation)"""
        # This is a simplified version
        # In practice, you would inflate all conv layers
        pass
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, T, H, W)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        # Reshape for 2D backbone: (B, C, T, H, W) -> (B*T, C, H, W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        
        # Extract features
        features = self.backbone(x)
        
        # Reshape back: (B*T, F) -> (B, T, F)
        features = features.view(B, T, -1)
        
        # Temporal pooling
        features = torch.mean(features, dim=1)
        
        # Classification
        features = self.dropout(features)
        logits = self.fc(features)
        
        return logits


def create_model(model_name: str, num_classes: int, config: Dict, pretrained: bool = True):
    """
    Create a video classification model with optional LoRA
    
    Args:
        model_name: Name of the model ('videomae', 'timesformer', 'i3d')
        num_classes: Number of output classes
        config: Model configuration dictionary
        pretrained: Whether to load pretrained weights
    
    Returns:
        model: PyTorch model (with LoRA if enabled)
    """
    if model_name.lower() == 'videomae':
        model = VideoMAEClassifier(num_classes, config, pretrained)
    elif model_name.lower() == 'timesformer':
        model = TimeSformerClassifier(num_classes, config, pretrained)
    elif model_name.lower() == 'i3d':
        model = I3DClassifier(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Apply LoRA if enabled
    if config.get('model', {}).get('use_lora', False):
        model = apply_lora(model, config)
        print(f"✅ LoRA applied! Trainable params: {count_parameters(model):,}")
    
    return model


def apply_lora(model, config: Dict):
    """
    Apply LoRA (Low-Rank Adaptation) to the model
    
    Args:
        model: PyTorch model
        config: Configuration dictionary with LoRA settings
    
    Returns:
        model: Model with LoRA applied
    """
    lora_config_dict = config.get('model', {}).get('lora', {})
    
    # Create LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        inference_mode=False,
        r=lora_config_dict.get('r', 8),
        lora_alpha=lora_config_dict.get('lora_alpha', 16),
        lora_dropout=lora_config_dict.get('lora_dropout', 0.1),
        target_modules=lora_config_dict.get('target_modules', ["qkv"]),
        bias=lora_config_dict.get('bias', "none"),
    )
    
    # Apply LoRA to the inner model (VideoMAE or TimeSformer)
    if hasattr(model, 'model'):
        model.model = get_peft_model(model.model, peft_config)
    else:
        model = get_peft_model(model, peft_config)
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🎯 LoRA Configuration:")
    print(f"   Rank (r): {peft_config.r}")
    print(f"   Alpha: {peft_config.lora_alpha}")
    print(f"   Target modules: {peft_config.target_modules}")
    print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"   Total params: {total_params:,}")
    
    return model


def load_checkpoint(model, checkpoint_path: str, device: str = 'cuda', config: Dict = None):
    """
    Load model from checkpoint (handles both regular and LoRA models)
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        config: Optional config dict for applying LoRA if needed
    
    Returns:
        model: Loaded model
        checkpoint: Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if checkpoint is from a PEFT model
    is_peft_checkpoint = checkpoint.get('is_peft_model', False)
    
    # If checkpoint is from PEFT model and config is provided, apply LoRA first
    if is_peft_checkpoint and config is not None:
        if config.get('use_lora', False):
            print("🔧 Applying LoRA before loading checkpoint...")
            model = apply_lora(model, config)
    
    # Load model state dict
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            print(f"⚠️  Warning: {e}")
            print("Trying non-strict loading...")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint)
    
    return model, checkpoint


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    checkpoint_path: str,
    **kwargs
):
    """
    Save model checkpoint (handles both regular and LoRA models)
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        best_metric: Best metric value
        checkpoint_path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    # Check if model is a PEFT model (LoRA)
    is_peft_model = hasattr(model, 'model') and hasattr(model.model, 'peft_config')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
        'is_peft_model': is_peft_model,
        **kwargs
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    # Also save LoRA adapter separately if using PEFT
    if is_peft_model:
        adapter_path = checkpoint_path.replace('.pth', '_lora_adapter')
        model.model.save_pretrained(adapter_path)
        print(f"💾 LoRA adapter saved to: {adapter_path}")


def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model, input_size=(1, 3, 16, 224, 224)):
    """
    Get model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, T, H, W)
    
    Returns:
        summary: Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'input_size': input_size
    }
    
    return summary


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether to minimize or maximize metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        """
        Check if training should be stopped
        
        Args:
            score: Current metric value
        
        Returns:
            early_stop: Boolean indicating whether to stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        """Check if score is an improvement"""
        if self.mode == 'min':
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)


def freeze_backbone(model, freeze: bool = True):
    """
    Freeze or unfreeze model backbone
    
    Args:
        model: PyTorch model
        freeze: Whether to freeze (True) or unfreeze (False)
    """
    if hasattr(model, 'model'):  # For wrapped models
        model = model.model
    
    if hasattr(model, 'videomae') or hasattr(model, 'timesformer'):
        # Freeze transformer backbone
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = not freeze
    else:
        # Freeze everything except last layer
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = not freeze


def get_optimizer(model, config: Dict):
    """
    Create optimizer
    
    Args:
        model: PyTorch model
        config: Training configuration
    
    Returns:
        optimizer: PyTorch optimizer
    """
    opt_config = config['training']['optimizer']
    name = opt_config['name'].lower()
    
    # Convert string values to float (safety check for YAML parsing issues)
    lr = float(opt_config['lr'])
    weight_decay = float(opt_config['weight_decay'])
    
    if name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=opt_config['betas'],
            weight_decay=weight_decay
        )
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=opt_config['betas'],
            weight_decay=weight_decay
        )
    elif name == 'sgd':
        momentum = float(opt_config['momentum'])
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizer


def get_scheduler(optimizer, config: Dict, num_training_steps: int):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration
        num_training_steps: Total number of training steps
    
    Returns:
        scheduler: PyTorch scheduler
    """
    sched_config = config['training']['scheduler']
    name = sched_config['name'].lower()
    
    if name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=float(sched_config['min_lr'])
        )
    elif name == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=int(sched_config['step_size']),
            gamma=float(sched_config['gamma'])
        )
    elif name == 'reduce_on_plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(sched_config['factor']),
            patience=int(sched_config['patience'])
        )
    else:
        scheduler = None
    
    return scheduler

