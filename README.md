# Member 2: Image Models & Explainable AI (XAI)
## Project: Real-time Groundwater Resource Evaluation using Image Processing and IoT
### Dataset: Zenodo (doi: 10.5281/zenodo.14537782) - Nagkoulis et al. (2025)

---

### 1. Role Overview
As **Member 2**, your responsibility is to analyze remote sensing / satellite / drone imagery to evaluate surface hydrological indicators, vegetation moisture stress, and groundwater depletion risk levels.

Your pipeline provides:
1. **Predicted Groundwater Depletion / Stress Class**:
   - `0_severe_depletion`: High risk / critically low groundwater table
   - `1_moderate_stress`: Medium risk / declining moisture
   - `2_safe_abundant`: Low risk / surface water present / high recharge
2. **Confidence Score / Probability Vector**: Fused with IoT sensor readings (depth, soil moisture) by Member 1/3.
3. **Explainable AI (XAI) Heatmaps & Attribution Masks**: Grad-CAM, LIME, and SHAP explaining *why* an area was flagged.

---

### 2. Vision Transformer & CNN Architectures

| Architecture | Paradigm | Specialization for Groundwater Imagery | Classification Head Design |
| :--- | :--- | :--- | :--- |
| **Swin Transformer** | Shifted Windows Transformer | Shifted local windows capture spatial features across adjacent agricultural plots with linear computational complexity. | Stage 4 features $\rightarrow$ LayerNorm $\rightarrow$ Dropout(0.3) $\rightarrow$ Linear(128) $\rightarrow$ Linear(num_classes) |
| **SegFormer** | Hierarchical Vision Transformer | Multi-scale Mix Transformer (MiT) models long-range spatial correlations across large satellite tiles without positional encoding artifacts. | MiT Encoder $\rightarrow$ LayerNorm $\rightarrow$ Global Avg Pool $\rightarrow$ Linear Classifier Head |
| **Attention U-Net** | Spatial Attention CNN | Attention Gates suppress cloud/urban clutter and focus on subtle vegetation and moisture boundaries. | Deep Bottleneck $\rightarrow$ Global Avg Pool $\rightarrow$ Linear(256, 128) $\rightarrow$ Dropout(0.4) $\rightarrow$ Linear(128, N) |
| **DeepLabV3+** | Dilated Conv / ASPP | Atrous Spatial Pyramid Pooling captures multi-scale geographic features (ponds, canals, wide arid basins). | ASPP feature map (256-dim) $\rightarrow$ Global Avg Pool $\rightarrow$ BatchNorm $\rightarrow$ Dropout(0.4) $\rightarrow$ Linear(128, N) |

---

### 3. Hyperparameter Tuning Protocol

| Hyperparameter | Evaluated Range | Recommended Setting | Rationale |
| :--- | :--- | :--- | :--- |
| **Optimizer** | AdamW vs. SGD | **AdamW** | Stable convergence on Vision Transformers (Swin / SegFormer) |
| **Learning Rate** | $10^{-5}$ to $10^{-3}$ | **$10^{-4}$ (Swin/SegFormer)**, **$3 \times 10^{-4}$ (CNNs)** | Avoids destabilizing pretrained weights |
| **Scheduler** | StepLR vs. CosineAnnealing | **CosineAnnealingLR (T_max=15)** | Smooth decay avoiding local minima |
| **Weight Decay** | $10^{-4}$ to $10^{-2}$ | **$10^{-4}$** | Regularizes high-capacity models without underfitting |
| **Augmentation** | Flips, ColorJitter, Rotations | **Random H/V Flips + Light Jitter** | Simulates varying sun angles and atmospheric conditions |

---

### 4. Running the Code

#### Option A: 1-Click Google Colab (Recommended for Zenodo 3.4 GB)
Open and run [`Groundwater_Models_XAI_Zenodo.ipynb`](Groundwater_Models_XAI_Zenodo.ipynb) in Google Colab on a free T4 GPU.

#### Option B: Local Training via CLI

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Train Swin Transformer
python train.py --model swin --epochs 15 --lr 0.0001 --optimizer adamw

# 3. Train SegFormer
python train.py --model segformer --epochs 15 --lr 0.00005 --optimizer adamw

# 4. Train Attention U-Net
python train.py --model attention_unet --epochs 15 --lr 0.0003 --optimizer adamw

# 5. Train DeepLabV3+
python train.py --model deeplabv3 --epochs 15 --lr 0.0001 --optimizer adamw
```

#### Step 5: Generate Explainable AI (XAI) Visualizations
```bash
python evaluate_and_explain.py --model swin
python evaluate_and_explain.py --model segformer
python evaluate_and_explain.py --model attention_unet
python evaluate_and_explain.py --model deeplabv3
```
Output comparison figures will be saved in `results/`.
