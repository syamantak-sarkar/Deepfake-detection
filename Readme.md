
# 🧠 Unsupervised and Generalizable Deepfake Detection Using Singular Value Decomposition

![Process Diagram](deepfake_detection_process.png) <!-- Replace with your actual image path -->

## 🔍 Overview

This project introduces an **unsupervised deepfake detection** pipeline based on **Singular Value Decomposition (SVD)**, aiming to identify deepfakes without relying on labeled training data. The method is designed to be **generalizable across datasets**, formats, and manipulation techniques.

The approach leverages the **spectral properties of image representations** to distinguish real from fake media, making it lightweight, interpretable, and adaptable.

---

## 🧪 Key Contributions

- ✅ **Unsupervised**: No need for annotated datasets.
- 🔁 **Generalizable**: Works across multiple deepfake datasets without retraining.
- 🔍 **SVD-Based**: Uses spectral signatures extracted via SVD for analysis.
- 📊 **Robust Evaluation**: Includes metrics, visualizations, and comparative benchmarks.

---

## 🛠️ Methodology

The process follows these main steps:

1. **Preprocessing** – Face detection and alignment.
2. **Feature Extraction** – Compute SVD of image regions or entire frames.
3. **Spectral Analysis** – Analyze singular values and vectors to detect anomalies.
4. **Decision** – Use unsupervised metrics to classify input as real or fake.

_See the diagram above for a visual summary._

---

## 📁 Folder Structure

<!-- Add this later after organizing the repo -->

---

## 📊 Experiments & Evaluation

### 📂 Cross-Dataset Evaluation (6 datasets)

### ✅ Model Trained on FF++-c23

Cross-dataset evaluations using the **frame-level ROC-AUC** metric. All detectors are trained on **FF++-c23** and evaluated on other datasets.

| Method    | Detector         | Backbone      | CDF-v1       | CDF-v2       | DFD          | DFDC         | DFDCP        | Avg.         |
|-----------|------------------|---------------|--------------|--------------|--------------|--------------|--------------|--------------|
| Naive     | Meso4            | MesoNet       | 0.736        | 0.609        | 0.548        | 0.556        | 0.599        | 0.610        |
| Naive     | MesoIncep        | MesoNet       | 0.737        | 0.697        | 0.623        | 0.576        | 0.684        | 0.663        |
| Naive     | CNN-Aug          | ResNet        | 0.742        | 0.703        | 0.646        | 0.636        | 0.617        | 0.669        |
| Naive     | Xception         | Xception      | 0.791        | 0.739        | 0.816        | 0.680        | 0.737        | 0.753        |
| Naive     | EfficientB4      | EfficientNet  | 0.791        | 0.749        | 0.815        | 0.696        | 0.728        | 0.756        |
| Spatial   | CapsuleNet       | Capsule       | 0.791        | 0.747        | 0.684        | 0.647        | 0.657        | 0.705        |
| Spatial   | FWA              | Xception      | 0.719        | 0.710        | 0.667        | 0.638        | 0.690        | 0.685        |
| Spatial   | Face X-ray       | HRNet         | 0.709        | 0.679        | 0.766        | 0.633        | 0.694        | 0.696        |
| Spatial   | FFD              | Xception      | 0.780        | 0.748        | 0.780        | 0.734        | 0.753        | 0.759        |
| Spatial   | CORE             | Xception      | 0.780        | 0.743        | 0.802        | 0.743        | 0.753        | 0.754        |
| Spatial   | Recce            | Custom        | 0.768        | -            | 0.812        | 0.713        | 0.734        | 0.752        |
| Spatial   | UCF              | Xception      | 0.779        | -            | 0.810        | 0.759        | 0.763        | 0.778        |
| Frequency | F3Net            | Xception      | 0.777        | 0.735        | 0.798        | 0.702        | 0.735        | 0.749        |
| Frequency | SPSL             | Xception      | 0.815        | 0.726        | 0.804        | 0.741        | 0.761        | 0.769        |
| Frequency | SRM              | Xception      | 0.793        | 0.755        | 0.812        | 0.704        | 0.741        | 0.760        |
| Frequency | EFNB4 + LSDA     | EfficientNet  | <u>0.867</u> | <u>0.830</u> | <u>0.880</u> | <u>0.736</u> | <u>0.815</u> | <u>0.826</u> |
| **SVD (Ours)** | U-Net VAE   | U-Net VAE     | <b>0.892</b> <span style="color:blue;">(+0.025)</span> | <b>0.876</b> <span style="color:blue;">(+0.046)</span> | <b>0.890</b> <span style="color:blue;">(+0.010)</span> | <b>0.834</b> <span style="color:blue;">(+0.098)</span> | <b>0.903</b> <span style="color:blue;">(+0.088)</span> | <b>0.881</b> <span style="color:blue;">(+0.055)</span> |

> Best results are in **bold**, second-best are <u>underlined</u>, and the improvements in ROC-AUC are shown in <span style="color:blue;">blue</span>.

#### ✅ Model Trained on CDF-v1

<!-- Paste next table -->

### 🧪 Cross-Manipulation Evaluation (4 types)
_Performance across four different deepfake manipulation techniques._

<!-- Details to be added -->

---

### 📈 Visualization of Reconstruction Loss (6 examples)
_Comparison of reconstruction loss between real and fake frames/images._

<!-- Visualizations and explanations go here -->

---

### 🎯 Effect of Threshold on ROC-AUC
_How varying the decision threshold affects ROC-AUC and detection robustness._

<!-- Add plots and analysis -->

---

## 🚀 To-Do

- [ ] Add dataset preprocessing instructions
- [ ] Include demo notebook or script
- [ ] Add trained model weights (if applicable)
- [ ] Finalize results and visualizations

---

## 🤝 Contributions

Feel free to open an issue or pull request for suggestions, improvements, or collaborations!

---

## 📜 License

MIT License

