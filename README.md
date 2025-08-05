# Abdominal Organ Segmentation Using MONAI

A deep learning pipeline for automated abdominal organ segmentation using the MONAI framework and 3D U-Net architecture. This project segments 13 different abdominal organs from CT scans using the FLARE22 dataset.

## 🎯 Overview

This implementation provides a complete end-to-end pipeline for medical image segmentation, achieving a **72.66% mean Dice score** on validation data. The model can segment the following abdominal organs:

- Liver
- Right Kidney
- Spleen
- Pancreas
- Aorta
- Inferior Vena Cava
- Right Adrenal Gland
- Left Adrenal Gland
- Gallbladder
- Esophagus
- Stomach
- Duodenum
- Left Kidney

## 🏗️ Architecture

- **Model**: 3D U-Net with residual connections
- **Framework**: MONAI (Medical Open Network for AI)
- **Input**: Single-channel CT scans
- **Output**: 14-channel segmentation maps (13 organs + background)
- **Patch Size**: 96×96×96 voxels
- **Loss Function**: Dice Loss with softmax activation

## 📋 Requirements

```python
# Core dependencies
torch
torchvision
monai[nibabel,tqdm]
numpy
matplotlib
```

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Install MONAI with required dependencies
pip install "monai[nibabel, tqdm]"
```

### 2. Data Preparation

- Organize your FLARE22 dataset as follows:
```
FLARE22Train/
├── images/
│   ├── FLARE22_Tr_0001_0000.nii
│   ├── FLARE22_Tr_0002_0000.nii
│   └── ...
└── labels/
    ├── FLARE22_Tr_0001.nii
    ├── FLARE22_Tr_0002.nii
    └── ...
```

### 3. Training

The pipeline automatically:
- Loads and preprocesses CT scans
- Applies data augmentations
- Trains the 3D U-Net model
- Validates performance every 2 epochs
- Saves the best model checkpoint

```python
# Key training parameters
max_epochs = 100
batch_size = 2
learning_rate = 1e-4
validation_interval = 2
```

## 🔧 Key Features

### Data Preprocessing
- **Resampling**: Standardizes voxel spacing to (1.5, 1.5, 2.0) mm
- **Intensity Normalization**: Clips HU values to [-175, 250] and normalizes to [0, 1]
- **Spatial Orientation**: Converts to RAS orientation
- **Patch Extraction**: Random crop-by-label for balanced training

### Data Augmentation
- **Rotation**: Random 90-degree rotations (10% probability)
- **Noise**: Gaussian noise injection (10% probability)
- **Spatial Augmentation**: Position/negative sampling for organ detection

### Training Strategy
- **Optimizer**: Adam with 1e-4 learning rate
- **Loss Function**: Dice Loss with one-hot encoding
- **Validation**: Sliding window inference with overlap
- **Metric**: Mean Dice coefficient across all organs

## 📊 Results

### Performance Metrics
- **Final Mean Dice Score**: 72.66%
- **Best Epoch**: 100
- **Training Duration**: 100 epochs
- **Validation Strategy**: 80/20 train-validation split

### Model Convergence
- Loss decreased from 0.97 to 0.56 over 100 epochs
- Dice score improved from 2.73% to 72.66%
- Consistent improvement with occasional plateaus

## 🔍 Evaluation & Visualization

The pipeline includes comprehensive evaluation:

1. **Per-case Analysis**: Identifies best and worst performing validation cases
2. **Visual Comparison**: Side-by-side comparison of ground truth vs. predictions
3. **Quantitative Metrics**: Dice coefficients for each validation case

### Sample Results
- **Best Case**: FLARE22_Tr_0050 (Dice: 81.71%)
- **Worst Case**: FLARE22_Tr_0045 (Dice: 64.47%)

## 💡 Key Improvements Implemented

1. **Enhanced Data Augmentation**: Added rotation and noise for better generalization
2. **Extended Training**: 100 epochs for better convergence
3. **Comprehensive Evaluation**: Best/worst case analysis with visualizations
4. **Robust Preprocessing**: Standardized pipeline for consistent results

## 🛠️ Technical Details

### Model Architecture
```python
UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=14,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
)
```

### Data Transforms
- **Training**: Includes augmentations for robustness
- **Validation**: Clean preprocessing without augmentations
- **Inference**: Sliding window with 50% overlap

## 📈 Usage Example

```python
# Load the trained model
model.load_state_dict(torch.load("best_metric_model.pth"))
model.eval()

# Perform inference on new CT scan
with torch.no_grad():
    prediction = sliding_window_inference(
        inputs, roi_size, sw_batch_size, model
    )
```

## 🔗 References

- [MONAI Framework](https://monai.io/)
- [FLARE22 Challenge](https://flare22.grand-challenge.org/)
- [3D U-Net Paper](https://arxiv.org/abs/1606.06650)

## 📝 License

This project is available for research and educational purposes. Please cite appropriately if used in academic work.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the implementation.

---

**Note**: This implementation requires significant computational resources (GPU recommended) and the FLARE22 dataset for training and evaluation.
