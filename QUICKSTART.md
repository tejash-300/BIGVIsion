# Quick Start Guide - Video Classification Project

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (2 min)

```bash
cd /home/ubuntu/tejash/video-classification-project
pip install -r requirements.txt
```

### Step 2: Prepare Your Dataset (1 min)

**Option A: Use UCF101 Dataset**
```bash
# Download from: https://www.crcv.ucf.edu/data/UCF101.php
# Extract to: data/raw/
```

**Option B: Use Your Own Videos**
```bash
# Organize your videos like this:
# data/raw/
# ├── class1/
# │   ├── video1.mp4
# │   └── video2.mp4
# ├── class2/
# │   └── ...
```

### Step 3: Run the Notebooks (2 min)

Open Jupyter Lab:
```bash
jupyter lab
```

Then execute notebooks in order:

1. **`01_data_preprocessing_and_exploration.ipynb`**
   - Loads and explores your dataset
   - Creates train/val/test splits
   - Visualizes data distribution

2. **`02_model_training.ipynb`**
   - Trains VideoMAE or TimeSformer model
   - Saves checkpoints automatically
   - Plots training curves

3. **`03_evaluation_and_visualization.ipynb`**
   - Evaluates model performance
   - Generates confusion matrix
   - Shows per-class metrics

4. **`04_inference_and_prediction.ipynb`**
   - Makes predictions on new videos
   - Exports results to JSON/CSV
   - Real-time webcam inference (optional)

---

## ⚙️ Quick Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Change model
model:
  name: "videomae"  # or "timesformer", "i3d"

# Adjust training
training:
  batch_size: 8     # Reduce if OOM
  num_epochs: 50    # Increase for better accuracy
  
# Modify video processing
video:
  num_frames: 16    # Number of frames to sample
  frame_size: [224, 224]  # Resolution
```

---

## 🎯 Expected Results

### Training Time
- **VideoMAE**: ~2-3 hours (100 epochs, single GPU)
- **TimeSformer**: ~2-3 hours
- **I3D**: ~1-2 hours

### Performance (UCF101)
- **Accuracy**: 85-95% depending on model and training
- **Top-5 Accuracy**: 95-99%

---

## 🐛 Troubleshooting

### Out of Memory?
```yaml
# In config.yaml, reduce:
training:
  batch_size: 4  # or even 2
video:
  num_frames: 8  # instead of 16
```

### Slow Data Loading?
```yaml
training:
  num_workers: 8  # Increase number of workers
```

### CUDA Not Available?
```yaml
hardware:
  device: "cpu"  # Switch to CPU mode
```

---

## 📊 Monitor Training

### TensorBoard (Optional)
```bash
tensorboard --logdir outputs/logs
```

### Weights & Biases (Optional)
```yaml
# In config.yaml:
logging:
  wandb:
    enabled: true
    project_name: "video-classification"
```

---

## 🎓 Next Steps

1. **Experiment with models**: Try VideoMAE, TimeSformer, and I3D
2. **Tune hyperparameters**: Learning rate, batch size, augmentation
3. **Add more data**: More training data = better performance
4. **Deploy your model**: Export to ONNX for production
5. **Try on webcam**: Real-time video classification!

---

## 💡 Tips for Best Results

1. **Use pretrained models**: Set `pretrained: true` in config
2. **Enable mixed precision**: Faster training, less memory
3. **Data augmentation**: Helps prevent overfitting
4. **Early stopping**: Prevents overfitting automatically
5. **Monitor validation accuracy**: Should improve over time

---

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Review notebook comments and markdown cells
- Inspect configuration options in `config.yaml`
- Look at utility functions in `utils/` directory

---

**Happy Training! 🚀**

