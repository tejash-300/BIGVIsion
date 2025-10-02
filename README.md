# Video Classification Project
## Big Vision Internship Assignment

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive video classification project using state-of-the-art transformer-based models (VideoMAE and TimeSformer) for action recognition on the UCF101 dataset.

---

## 🎯 Project Overview

This project implements modern deep learning approaches for video classification, featuring:

- **State-of-the-art Models**: VideoMAE, TimeSformer, and I3D
- **Efficient Data Pipeline**: Optimized video loading and preprocessing
- **Advanced Training**: Mixed precision training, gradient accumulation, label smoothing
- **Comprehensive Evaluation**: Multiple metrics, confusion matrices, per-class analysis
- **Beautiful Visualizations**: Training curves, predictions, attention maps
- **Production-Ready**: Model export, inference pipeline, deployment ready

---

## 📁 Project Structure

```
video-classification-project/
├── configs/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                     # Raw video data
│   ├── processed/               # Preprocessed data
│   └── splits/                  # Train/val/test splits
├── models/                      # Saved model checkpoints
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_preprocessing_and_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation_and_visualization.ipynb
│   └── 04_inference_and_prediction.ipynb
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── data_utils.py           # Data loading and processing
│   ├── model_utils.py          # Model creation and management
│   ├── training_utils.py       # Training and evaluation functions
│   └── visualization_utils.py  # Visualization functions
├── outputs/                     # Output files
│   ├── logs/                   # Training logs
│   ├── visualizations/         # Generated plots
│   └── predictions/            # Prediction results
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/video-classification-project.git
cd video-classification-project
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Download Dataset

Download the UCF101 dataset:
```bash
# Visit: https://www.crcv.ucf.edu/data/UCF101.php
# Download and extract to data/raw/
```

### 4. Run Notebooks

Execute notebooks in order:

1. **Data Preprocessing**: `01_data_preprocessing_and_exploration.ipynb`
2. **Model Training**: `02_model_training.ipynb`
3. **Evaluation**: `03_evaluation_and_visualization.ipynb`
4. **Inference**: `04_inference_and_prediction.ipynb`

---

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

- **Dataset**: UCF101 or custom dataset
- **Model**: VideoMAE, TimeSformer, or I3D
- **Training**: Batch size, learning rate, epochs
- **Augmentation**: Data augmentation strategies
- **Hardware**: GPU configuration

Example:
```yaml
model:
  name: "videomae"  # Options: videomae, timesformer, i3d
  pretrained: true

training:
  batch_size: 8
  num_epochs: 100
  optimizer:
    name: "adamw"
    lr: 1e-4
```

---

## 📊 Models

### VideoMAE
- **Architecture**: Vision Transformer for videos
- **Pretraining**: Masked autoencoding
- **Performance**: State-of-the-art on UCF101
- **Paper**: [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)

### TimeSformer
- **Architecture**: Divided space-time attention
- **Approach**: Separate spatial and temporal attention
- **Performance**: Efficient and accurate
- **Paper**: [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)

### I3D
- **Architecture**: Inflated 3D ConvNets
- **Approach**: 2D to 3D inflation
- **Performance**: Strong baseline
- **Paper**: [Quo Vadis, Action Recognition?](https://arxiv.org/abs/1705.07750)

---

## 📈 Results

Expected performance on UCF101:

| Model | Accuracy | Top-5 Acc | Parameters | FPS |
|-------|----------|-----------|------------|-----|
| VideoMAE | ~93%+ | ~99%+ | 86M | 15 |
| TimeSformer | ~90%+ | ~98%+ | 121M | 20 |
| I3D | ~88%+ | ~97%+ | 25M | 30 |

*Note: Results may vary based on training configuration*

---

## 🎓 Key Features

### Data Processing
- ✅ Efficient video loading with Decord
- ✅ Smart frame sampling (uniform, random, dense)
- ✅ Data augmentation pipeline
- ✅ Stratified train/val/test splits

### Training
- ✅ Mixed precision training (FP16)
- ✅ Gradient clipping and accumulation
- ✅ Label smoothing
- ✅ Cosine learning rate scheduling
- ✅ Early stopping
- ✅ Model checkpointing

### Evaluation
- ✅ Accuracy, Top-5 accuracy
- ✅ Precision, Recall, F1-score
- ✅ Confusion matrix
- ✅ Per-class metrics
- ✅ Classification reports

### Visualization
- ✅ Training curves
- ✅ Confusion matrices
- ✅ Sample predictions
- ✅ Class distributions
- ✅ Attention visualizations

---

## 💡 Advanced Features

### Mixed Precision Training
Automatically enabled for faster training and reduced memory usage:
```yaml
training:
  mixed_precision: true
```

### Data Augmentation
Comprehensive augmentation pipeline:
```yaml
augmentation:
  train:
    random_crop: true
    horizontal_flip: true
    color_jitter: true
    rotation_range: 15
```

### Learning Rate Scheduling
Multiple scheduler options:
```yaml
scheduler:
  name: "cosine"  # cosine, step, reduce_on_plateau
  warmup_epochs: 5
  min_lr: 1e-6
```

---

## 📝 Usage Examples

### Training a Model

```python
# In 02_model_training.ipynb
from utils.model_utils import create_model, get_optimizer, get_scheduler
from utils.training_utils import train_one_epoch, validate

# Create model
model = create_model('videomae', num_classes=101, config=config)

# Train
for epoch in range(num_epochs):
    train_metrics = train_one_epoch(model, train_loader, ...)
    val_metrics = validate(model, val_loader, ...)
```

### Making Predictions

```python
# In 04_inference_and_prediction.ipynb
from utils.model_utils import load_checkpoint

# Load model
model, checkpoint = load_checkpoint(model, 'models/best_model.pth')

# Predict
outputs = model(video_tensor)
predictions = outputs.argmax(dim=1)
```

---

## 🔍 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce batch size in `config.yaml`
- Enable gradient accumulation
- Use mixed precision training

**2. Slow Training**
- Check data loading bottleneck
- Increase `num_workers`
- Use smaller frame resolution

**3. Poor Performance**
- Increase training epochs
- Try different learning rates
- Add more data augmentation
- Use pretrained models

---

## 📚 References

1. **VideoMAE**: [Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)
2. **TimeSformer**: [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
3. **UCF101 Dataset**: [UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/abs/1212.0402)
4. **Vision Transformers**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Big Vision for the internship opportunity
- HuggingFace Transformers for pretrained models
- PyTorch team for the excellent framework
- UCF for the UCF101 dataset

---

## 📊 Project Status

- [x] Data preprocessing pipeline
- [x] Model implementation
- [x] Training pipeline
- [x] Evaluation metrics
- [x] Visualization tools
- [ ] Model deployment
- [ ] Web interface
- [ ] Real-time inference

---

**Built with ❤️ for video understanding**

