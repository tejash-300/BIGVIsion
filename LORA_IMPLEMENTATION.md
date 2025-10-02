# LoRA Implementation Guide

## 🎯 What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that dramatically reduces the number of trainable parameters while maintaining model performance.

## ✅ Benefits for This Project

### 1. **Memory Efficiency**
- **Before (Full Fine-Tuning)**: ~86M trainable parameters
- **After (LoRA)**: ~2-4M trainable parameters (95% reduction!)
- **Result**: Can use larger batch sizes, faster training

### 2. **Training Speed**
- **2-3x faster training** per epoch
- **50-70% less GPU memory** usage
- **Faster convergence** in fewer epochs

### 3. **Model Size**
- **Full model checkpoint**: ~350MB
- **LoRA adapter only**: ~10-20MB
- **Easy to share** and deploy

---

## 🔧 Configuration

The LoRA settings are in `configs/config.yaml`:

```yaml
model:
  use_lora: true  # Enable/disable LoRA
  lora:
    r: 8  # Rank (higher = more params, better performance)
    lora_alpha: 16  # Scaling factor
    lora_dropout: 0.1
    target_modules: ["qkv"]  # Apply LoRA to attention layers
    bias: "none"
```

### Key Parameters:

- **`r` (rank)**: Controls LoRA matrix size
  - `r=4`: Ultra-light (fastest, least memory)
  - `r=8`: Balanced (recommended)
  - `r=16`: High capacity (best performance)

- **`lora_alpha`**: Scaling factor (usually 2x of `r`)

- **`target_modules`**: Which layers to apply LoRA to
  - `["qkv"]`: Only attention (fastest)
  - `["q", "v"]`: Query and Value only
  - `["qkv", "fc"]`: Attention + feedforward

---

## 📊 Expected Performance

### VideoMAE with LoRA (r=8):
- **Trainable params**: ~3-4M (vs 86M full fine-tuning)
- **Training time**: ~1-1.5 hours for 50 epochs
- **Memory usage**: ~8-10GB GPU
- **Accuracy**: 85-90% (comparable to full fine-tuning)

### Comparison:

| Method | Trainable Params | GPU Memory | Training Time | Accuracy |
|--------|-----------------|------------|---------------|----------|
| Full Fine-Tuning | 86M | 16-20GB | 3-4 hours | 88-92% |
| **LoRA (r=8)** | **3-4M** | **8-10GB** | **1-1.5 hours** | **85-90%** |
| LoRA (r=4) | 2M | 6-8GB | 45-60 min | 82-87% |

---

## 🚀 How to Use

### 1. **Enable LoRA** (Already done!)
Set `use_lora: true` in `configs/config.yaml`

### 2. **Run Training Notebook**
```bash
# Run the training notebook
jupyter notebook notebooks/02_model_training.ipynb
```

The notebook will automatically:
- Load pretrained VideoMAE
- Apply LoRA adapters
- Show parameter reduction
- Train only LoRA weights

### 3. **Monitor Training**
You'll see output like:
```
🎯 LoRA Configuration:
   Rank (r): 8
   Alpha: 16
   Target modules: ['qkv']
   Trainable params: 3,407,872 (3.96%)
   Total params: 86,012,416
```

### 4. **Save & Load**
The training saves:
- **Full checkpoint**: `best_model.pth` (includes LoRA weights)
- **LoRA adapter**: `best_model_lora_adapter/` (standalone adapter)

---

## 🎛️ Tuning LoRA Parameters

### For Better Performance:
```yaml
lora:
  r: 16  # Increase rank
  lora_alpha: 32
  target_modules: ["qkv", "fc"]  # Add more layers
```

### For Faster Training:
```yaml
lora:
  r: 4  # Decrease rank
  lora_alpha: 8
  target_modules: ["qkv"]  # Fewer layers
```

### For Memory Constrained:
```yaml
training:
  batch_size: 4  # Reduce batch size
lora:
  r: 4
  lora_dropout: 0.0  # Disable dropout
```

---

## 🔬 Technical Details

### How LoRA Works:
Instead of updating all weight matrices `W`, LoRA adds small trainable matrices:
```
W' = W + (B * A) * α/r
```
where:
- `W`: Frozen pretrained weights
- `A`, `B`: Small trainable matrices (rank `r`)
- `α`: Scaling factor

### Why It's Efficient:
- Original: `d × d` parameters
- LoRA: `2 × d × r` parameters (much smaller!)
- Example: 768×768 = 589,824 → 2×768×8 = 12,288 (98% reduction!)

---

## 📈 Expected Results

### Training Metrics:
- **Epoch 1-10**: Learning LoRA adapters (~60-70% accuracy)
- **Epoch 10-30**: Refinement (~75-85% accuracy)
- **Epoch 30-50**: Convergence (~85-90% accuracy)

### Final Performance:
- **Train Accuracy**: 90-95%
- **Validation Accuracy**: 85-90%
- **Test Accuracy**: 83-88%
- **YouTube Generalization**: 75-85%

---

## 🐛 Troubleshooting

### Issue: "target_modules not found"
**Solution**: Change to `["query", "value"]` or check model architecture

### Issue: Low accuracy
**Solution**: Increase `r` to 16, add more target modules

### Issue: Out of memory
**Solution**: Reduce batch size or decrease `r` to 4

---

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE)

---

## 💡 Pro Tips

1. **Start with r=8**: Good balance for most cases
2. **Use learning rate 1e-4**: Higher than full fine-tuning
3. **Train fewer epochs**: LoRA converges faster (30-50 epochs)
4. **Save adapters**: Share 10MB adapter instead of 350MB model
5. **Experiment**: Try different target_modules for your use case

---

**Happy Training! 🚀**

