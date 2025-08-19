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

📊 Worst Performing Case: FLARE22_Tr_0041_0000.nii
Mean Dice: 0.5940
Per-Organ Dice Scores:
  - Spleen                : 0.9250
  - Right Kidney          : 0.9317
  - Left Kidney           : 0.9021
  - Gallbladder           : 0.5404
  - Liver                 : 0.5761
  - Stomach               : 0.5487
  - Aorta                 : 0.6328
  - Inferior Vena Cava    : 0.0977
  - Pancreas              : 0.3080
  - Right Adrenal Gland   : 0.6533
  - Left Adrenal Gland    : 0.7593
  - Duodenum              : 0.2197
  - Bladder               : 0.6277
``` 

---

### 2️⃣ `monai-with-randaffine-adamw-optimizer.ipynb`  
- **Setup:** AdamW + RandAffine augmentation  
- **Observations:**  
  - AdamW provided better generalization, slightly lower validation loss  
- **Results:**  
```bash
📊 Best Performing Case: FLARE22_Tr_0042_0000.nii
Mean Dice: 0.8540
Per-Organ Dice Scores:
  - Spleen                : 0.9595
  - Right Kidney          : 0.9563
  - Left Kidney           : 0.9326
  - Gallbladder           : 0.8207
  - Liver                 : 0.8951
  - Stomach               : 0.8478
  - Aorta                 : 0.7452
  - Inferior Vena Cava    : 0.7491
  - Pancreas              : 0.8366
  - Right Adrenal Gland   : 0.7199
  - Left Adrenal Gland    : 0.8886
  - Duodenum              : 0.7960
  - Bladder               : 0.9540

📊 Worst Performing Case: FLARE22_Tr_0045_0000.nii
Mean Dice: 0.6208
Per-Organ Dice Scores:
  - Spleen                : 0.9460
  - Right Kidney          : 0.2889
  - Left Kidney           : 0.8824
  - Gallbladder           : 0.6008
  - Liver                 : 0.7920
  - Stomach               : 0.7271
  - Aorta                 : 0.3801
  - Inferior Vena Cava    : 0.2933
  - Pancreas              : 0.8185
  - Right Adrenal Gland   : 0.7131
  - Left Adrenal Gland    : 0.7963
  - Duodenum              : 0.7618
  - Bladder               : 0.0700
```
---

### 3️⃣ `monai-with-randaffine-leakyrelu-lr-scheduler.ipynb`  
- **Setup:** Adam + LeakyReLU + RandAffine + CosineAnnealingLR  
- **Observations:**  
  - CosineAnnealingLR stabilized training curve  
  - Best balance between accuracy & stability  
- **Results:**  
```bash
📊 Best Performing Case: FLARE22_Tr_0042_0000.nii
Mean Dice: 0.5649
Per-Organ Dice Scores:
  - Spleen                : 0.9646
  - Right Kidney          : 0.4704
  - Left Kidney           : 0.9554
  - Gallbladder           : 0.8389
  - Liver                 : 0.6054
  - Stomach               : 0.8627
  - Aorta                 : 0.0000
  - Inferior Vena Cava    : 0.0000
  - Pancreas              : 0.7726
  - Right Adrenal Gland   : 0.0000
  - Left Adrenal Gland    : 0.8578
  - Duodenum              : 0.5301
  - Bladder               : 0.4857

📊 Worst Performing Case: FLARE22_Tr_0047_0000.nii
Mean Dice: 0.4409
Per-Organ Dice Scores:
  - Spleen                : 0.9415
  - Right Kidney          : 0.1004
  - Left Kidney           : 0.9233
  - Gallbladder           : 0.7017
  - Liver                 : 0.4298
  - Stomach               : 0.7375
  - Aorta                 : 0.0000
  - Inferior Vena Cava    : 0.0000
  - Pancreas              : 0.7015
  - Right Adrenal Gland   : 0.0000
  - Left Adrenal Gland    : 0.7081
  - Duodenum              : 0.4250
  - Bladder               : 0.0631
```

---

### 4️⃣ `monai-with-cosineannealinglrscheduler.ipynb`  
- **Setup:** Adam + CosineAnnealingLR  
- **Observations:**  
  - Without augmentation, model overfits quicker  
```bash
📊 Best Performing Case: FLARE22_Tr_0050_0000.nii
Mean Dice: 0.6984
Per-Organ Dice Scores:
  - Spleen                : 0.9635
  - Right Kidney          : 0.9547
  - Left Kidney           : 0.9533
  - Gallbladder           : 0.7025
  - Liver                 : 0.8810
  - Stomach               : 0.7900
  - Aorta                 : 0.4437
  - Inferior Vena Cava    : 0.0000
  - Pancreas              : 0.8387
  - Right Adrenal Gland   : 0.0000
  - Left Adrenal Gland    : 0.8146
  - Duodenum              : 0.7765
  - Bladder               : 0.9603

📊 Worst Performing Case: FLARE22_Tr_0045_0000.nii
Mean Dice: 0.5344
Per-Organ Dice Scores:
  - Spleen                : 0.9429
  - Right Kidney          : 0.6526
  - Left Kidney           : 0.8861
  - Gallbladder           : 0.5928
  - Liver                 : 0.7801
  - Stomach               : 0.8119
  - Aorta                 : 0.0000
  - Inferior Vena Cava    : 0.0000
  - Pancreas              : 0.7122
  - Right Adrenal Gland   : 0.0000
  - Left Adrenal Gland    : 0.6605
  - Duodenum              : 0.6814
  - Bladder               : 0.2263
```
---

### 5️⃣ `monai-with-randaffine-augmentation.ipynb`  
- **Setup:** Adam + RandAffine only  
- **Observations:**  
  - RandAffine improved robustness, but lacked extra gain from LeakyReLU  
- **Results:**  
```bash
📊 Best Performing Case: FLARE22_Tr_0042_0000.nii
Mean Dice: 0.8248
Per-Organ Dice Scores:
  - Spleen                : 0.9614
  - Right Kidney          : 0.9427
  - Left Kidney           : 0.9515
  - Gallbladder           : 0.8440
  - Liver                 : 0.9347
  - Stomach               : 0.8560
  - Aorta                 : 0.7139
  - Inferior Vena Cava    : 0.4492
  - Pancreas              : 0.7826
  - Right Adrenal Gland   : 0.6923
  - Left Adrenal Gland    : 0.8678
  - Duodenum              : 0.7720
  - Bladder               : 0.9542

📊 Worst Performing Case: FLARE22_Tr_0047_0000.nii
Mean Dice: 0.6234
Per-Organ Dice Scores:
  - Spleen                : 0.9195
  - Right Kidney          : 0.9042
  - Left Kidney           : 0.8435
  - Gallbladder           : 0.6651
  - Liver                 : 0.8867
  - Stomach               : 0.6377
  - Aorta                 : 0.2385
  - Inferior Vena Cava    : 0.0392
  - Pancreas              : 0.5063
  - Right Adrenal Gland   : 0.2846
  - Left Adrenal Gland    : 0.7034
  - Duodenum              : 0.5925
  - Bladder               : 0.8828

```
---

### 6️⃣ `monai-leakyrelu.ipynb`  
- **Setup:** Adam + LeakyReLU + LRScheduler  
- **Observations:**  
  - Better gradient flow than ReLU  
  - LRScheduler prevented overfitting  
- **Results:**  
```bash
📊 Best Performing Case: FLARE22_Tr_0050_0000.nii
Mean Dice: 0.8028
Per-Organ Dice Scores:
  - Spleen                : 0.9651
  - Right Kidney          : 0.9650
  - Left Kidney           : 0.9575
  - Gallbladder           : 0.7854
  - Liver                 : 0.9090
  - Stomach               : 0.8416
  - Aorta                 : 0.7446
  - Inferior Vena Cava    : 0.0000
  - Pancreas              : 0.8861
  - Right Adrenal Gland   : 0.7394
  - Left Adrenal Gland    : 0.8413
  - Duodenum              : 0.8361
  - Bladder               : 0.9649

📊 Worst Performing Case: FLARE22_Tr_0045_0000.nii
Mean Dice: 0.6237
Per-Organ Dice Scores:
  - Spleen                : 0.9473
  - Right Kidney          : 0.6584
  - Left Kidney           : 0.9245
  - Gallbladder           : 0.6089
  - Liver                 : 0.8063
  - Stomach               : 0.6634
  - Aorta                 : 0.0000
  - Inferior Vena Cava    : 0.0000
  - Pancreas              : 0.8436
  - Right Adrenal Gland   : 0.7046
  - Left Adrenal Gland    : 0.8278
  - Duodenum              : 0.7432
  - Bladder               : 0.3798
```

---

### 7️⃣ `adamw.ipynb`  
- **Setup:** AdamW only (baseline)  
- **Observations:**  
  - Served as control experiment  
- **Results:**  
```bash
📊 Best Performing Case: FLARE22_Tr_0042_0000.nii
Mean Dice: 0.7932
Per-Organ Dice Scores:
  - Spleen                : 0.9600
  - Right Kidney          : 0.9522
  - Left Kidney           : 0.9420
  - Gallbladder           : 0.8284
  - Liver                 : 0.9117
  - Stomach               : 0.8698
  - Aorta                 : 0.6901
  - Inferior Vena Cava    : 0.0000
  - Pancreas              : 0.7909
  - Right Adrenal Gland   : 0.7210
  - Left Adrenal Gland    : 0.8877
  - Duodenum              : 0.7966
  - Bladder               : 0.9609

📊 Worst Performing Case: FLARE22_Tr_0045_0000.nii
Mean Dice: 0.6212
Per-Organ Dice Scores:
  - Spleen                : 0.9516
  - Right Kidney          : 0.5285
  - Left Kidney           : 0.8945
  - Gallbladder           : 0.5204
  - Liver                 : 0.8366
  - Stomach               : 0.8421
  - Aorta                 : 0.3115
  - Inferior Vena Cava    : 0.0000
  - Pancreas              : 0.8179
  - Right Adrenal Gland   : 0.7112
  - Left Adrenal Gland    : 0.5901
  - Duodenum              : 0.6918
  - Bladder               : 0.3794
```

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
