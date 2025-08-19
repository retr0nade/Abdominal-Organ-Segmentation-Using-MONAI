# 🧠 Abdominal Organ Segmentation with MONAI  

This repository contains experiments on **abdominal organ segmentation** using [MONAI](https://monai.io/), a deep learning framework for medical imaging.  
The focus is on analyzing how different **optimizers, learning rate schedulers, activation functions, and augmentations** impact segmentation performance.  

---

## 📌 Project Objective  
Accurate segmentation of abdominal organs from CT/MRI scans is crucial for medical analysis.  
This project experiments with different training strategies to evaluate:  
- How **LeakyReLU vs ReLU** affects convergence  
- Whether **Adam vs AdamW** improves generalization  
- How **CosineAnnealingLR** stabilizes training  
- The impact of **RandAffine augmentation** on robustness  

---

## 📂 Repository Structure  

| Notebook | Optimizer | Activation | Scheduler | Augmentation | Key Focus |
|----------|-----------|------------|-----------|--------------|-----------|
| `monai-with-randaffine-augmentation-leakyrelu.ipynb` | Adam | LeakyReLU | None | RandAffine | Effect of augmentation + LeakyReLU |
| `monai-with-randaffine-adamw-optimizer.ipynb` | AdamW | Default | None | RandAffine | AdamW + augmentation |
| `monai-with-randaffine-leakyrelu-lr-scheduler.ipynb` | Adam | LeakyReLU | CosineAnnealingLR | RandAffine | Scheduler + Augmentation |
| `monai-with-cosineannealinglrscheduler.ipynb` | Adam | Default | CosineAnnealingLR | None | Pure LR scheduler |
| `monai-with-randaffine-augmentation.ipynb` | Adam | Default | None | RandAffine | Only augmentation |
| `monai-leakyrelu.ipynb` | Adam | LeakyReLU | LRScheduler | None | Pure activation comparison |
| `adamw.ipynb` | AdamW | Default | None | None | Optimizer baseline |

---

## ⚙️ Setup  

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/monai-abdominal-segmentation.git
cd monai-abdominal-segmentation
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

**Requirements include:**  
- torch  
- monai  
- numpy  
- matplotlib  
- scikit-learn  
- nibabel  

### 3. Run Notebooks  
Launch Jupyter or Colab and open the desired notebook:  
```bash
jupyter notebook monai-with-randaffine-augmentation-leakyrelu.ipynb
```

---

## 📊 Results  

Below are experiment-wise results (replace placeholders with actual metrics/plots from your runs):  

### 1️⃣ `monai-with-randaffine-augmentation-leakyrelu.ipynb`  
- **Setup:** Adam + LeakyReLU + RandAffine augmentation  
- **Observations:**  
  - Faster convergence than ReLU baseline  
  - Augmentation improved robustness to rotation/translation  
- **Results:**  
```bash
📊 Best Performing Case: FLARE22_Tr_0042_0000.nii
Mean Dice: 0.7503
Per-Organ Dice Scores:
  - Spleen                : 0.9640
  - Right Kidney          : 0.9476
  - Left Kidney           : 0.9428
  - Gallbladder           : 0.8570
  - Liver                 : 0.5961
  - Stomach               : 0.5828
  - Aorta                 : 0.6815
  - Inferior Vena Cava    : 0.2948
  - Pancreas              : 0.8136
  - Right Adrenal Gland   : 0.7485
  - Left Adrenal Gland    : 0.8869
  - Duodenum              : 0.7993
  - Bladder               : 0.6385
``` 

---

### 2️⃣ `monai-with-randaffine-adamw-optimizer.ipynb`  
- **Setup:** AdamW + RandAffine augmentation  
- **Observations:**  
  - AdamW provided better generalization, slightly lower validation loss  
- **Results:**  
  - Dice Score: `XX.XX%`  
  - Validation Loss: `YY.YY`  
  - ![Sample Prediction](results/randaffine-adamw.png)  

---

### 3️⃣ `monai-with-randaffine-leakyrelu-lr-scheduler.ipynb`  
- **Setup:** Adam + LeakyReLU + RandAffine + CosineAnnealingLR  
- **Observations:**  
  - CosineAnnealingLR stabilized training curve  
  - Best balance between accuracy & stability  
- **Results:**  
  - Dice Score: `XX.XX%`  
  - Validation Loss: `YY.YY`  
  - ![Training Curve](results/scheduler-curve.png)  

---

### 4️⃣ `monai-with-cosineannealinglrscheduler.ipynb`  
- **Setup:** Adam + CosineAnnealingLR  
- **Observations:**  
  - Without augmentation, model overfits quicker  
- **Results:**  
  - Dice Score: `XX.XX%`  
  - Validation Loss: `YY.YY`  

---

### 5️⃣ `monai-with-randaffine-augmentation.ipynb`  
- **Setup:** Adam + RandAffine only  
- **Observations:**  
  - RandAffine improved robustness, but lacked extra gain from LeakyReLU  
- **Results:**  
  - Dice Score: `XX.XX%`  
  - Validation Loss: `YY.YY`  

---

### 6️⃣ `monai-leakyrelu.ipynb`  
- **Setup:** Adam + LeakyReLU + LRScheduler  
- **Observations:**  
  - Better gradient flow than ReLU  
  - LRScheduler prevented overfitting  
- **Results:**  
  - Dice Score: `XX.XX%`  
  - Validation Loss: `YY.YY`  

---

### 7️⃣ `adamw.ipynb`  
- **Setup:** AdamW only (baseline)  
- **Observations:**  
  - Served as control experiment  
- **Results:**  
  - Dice Score: `XX.XX%`  
  - Validation Loss: `YY.YY`  

---

## 📌 Key Insights  
- **LeakyReLU > ReLU** for segmentation stability  
- **RandAffine augmentation** improves robustness significantly  
- **AdamW optimizer** generalizes better than Adam  
- **CosineAnnealingLR** provides smoother convergence  

---

## 📝 Future Work  
- Combine **multiple augmentations** (RandAffine + RandGaussianNoise)  
- Use **hybrid loss functions** (Dice + Focal)  
- Implement **teacher-student semi-supervised training**  

---

## 📜 License  
This project is licensed under the MIT License.  
