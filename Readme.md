
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


#### ✅ Model Trained on CDF-v2

Cross-dataset evaluations using the **frame-level ROC-AUC** metric. All detectors are trained on **FF++** and evaluated on other datasets.

| Method    | Detector         | Backbone      | CDF-v1       | FF++         | DFD          | DFDC         | DFDCP        | Avg.         |
|-----------|------------------|---------------|--------------|--------------|--------------|--------------|--------------|--------------|
| Naive     | Meso4            | MesoNet       | 0.701        | 0.679        | 0.577        | 0.541        | 0.603        | 0.620        |
| Naive     | MesoIncep        | MesoNet       | 0.723        | 0.707        | 0.620        | 0.563        | 0.678        | 0.658        |
| Naive     | CNN-Aug          | ResNet        | 0.735        | 0.720        | 0.645        | 0.627        | 0.618        | 0.669        |
| Naive     | Xception         | Xception      | 0.776        | 0.758        | 0.813        | 0.687        | 0.740        | 0.755        |
| Naive     | EfficientB4      | EfficientNet  | 0.780        | 0.762        | 0.810        | 0.705        | 0.729        | 0.757        |
| Spatial   | CapsuleNet       | Capsule       | 0.780        | 0.759        | 0.688        | 0.650        | 0.661        | 0.708        |
| Spatial   | FWA              | Xception      | 0.710        | 0.720        | 0.675        | 0.638        | 0.692        | 0.687        |
| Spatial   | Face X-ray       | HRNet         | 0.696        | 0.682        | 0.768        | 0.637        | 0.695        | 0.696        |
| Spatial   | FFD              | Xception      | 0.763        | 0.758        | 0.785        | 0.736        | 0.755        | 0.759        |
| Spatial   | CORE             | Xception      | 0.765        | 0.751        | 0.807        | 0.744        | 0.755        | 0.764        |
| Spatial   | Recce            | Custom        | 0.755        | -            | 0.816        | 0.715        | 0.738        | 0.756        |
| Spatial   | UCF              | Xception      | 0.771        | -            | 0.814        | 0.765        | 0.760        | 0.778        |
| Frequency | F3Net            | Xception      | 0.760        | 0.745        | 0.800        | 0.706        | 0.735        | 0.749        |
| Frequency | SPSL             | Xception      | 0.799        | 0.736        | 0.804        | 0.749        | 0.767        | 0.771        |
| Frequency | SRM              | Xception      | 0.782        | 0.764        | 0.818        | 0.710        | 0.742        | 0.763        |
| Frequency | EFNB4 + LSDA     | EfficientNet  | <u>0.857</u> | <u>0.838</u> | <u>0.882</u> | <u>0.744</u> | <u>0.818</u> | <u>0.827</u> |
| **SVD (Ours)** | U-Net VAE   | U-Net VAE     | **0.881** <span style="color:blue;">(+0.024)</span> | **0.896** <span style="color:blue;">(+0.058)</span> | **0.899** <span style="color:blue;">(+0.017)</span> | **0.839** <span style="color:blue;">(+0.095)</span> | **0.910** <span style="color:blue;">(+0.092)</span> | **0.885** <span style="color:blue;">(+0.058)</span> |

### ✅ Model Trained on **CDF-v1**

Cross-dataset evaluations using the **frame-level ROC-AUC** metric. All detectors are trained on **CDF-v1** and evaluated on other datasets.

| Method    | Detector         | Backbone      | CDF-v2       | FF++         | DFD          | DFDC         | DFDCP        | Avg.         |
|-----------|------------------|---------------|--------------|--------------|--------------|--------------|--------------|--------------|
| Naive     | Meso4            | MesoNet       | 0.715        | 0.662        | 0.569        | 0.535        | 0.598        | 0.616        |
| Naive     | MesoIncep        | MesoNet       | 0.731        | 0.688        | 0.613        | 0.550        | 0.670        | 0.650        |
| Naive     | CNN-Aug          | ResNet        | 0.740        | 0.702        | 0.628        | 0.620        | 0.605        | 0.659        |
| Naive     | Xception         | Xception      | 0.781        | 0.740        | 0.795        | 0.675        | 0.729        | 0.744        |
| Naive     | EfficientB4      | EfficientNet  | 0.783        | 0.752        | 0.801        | 0.699        | 0.715        | 0.750        |
| Spatial   | CapsuleNet       | Capsule       | 0.769        | 0.742        | 0.678        | 0.642        | 0.648        | 0.696        |
| Spatial   | FWA              | Xception      | 0.722        | 0.705        | 0.661        | 0.625        | 0.684        | 0.679        |
| Spatial   | Face X-ray       | HRNet         | 0.709        | 0.668        | 0.762        | 0.626        | 0.690        | 0.691        |
| Spatial   | FFD              | Xception      | 0.765        | 0.746        | 0.775        | 0.725        | 0.745        | 0.751        |
| Spatial   | CORE             | Xception      | 0.769        | 0.738        | 0.790        | 0.735        | 0.748        | 0.756        |
| Spatial   | Recce            | Custom        | 0.750        | -            | 0.806        | 0.704        | 0.729        | 0.747        |
| Spatial   | UCF              | Xception      | 0.765        | -            | 0.804        | 0.754        | 0.749        | 0.768        |
| Frequency | F3Net            | Xception      | 0.751        | 0.736        | 0.795        | 0.700        | 0.720        | 0.740        |
| Frequency | SPSL             | Xception      | 0.789        | 0.727        | 0.798        | 0.742        | 0.760        | 0.763        |
| Frequency | SRM              | Xception      | 0.774        | 0.756        | 0.809        | 0.705        | 0.735        | 0.756        |
| Frequency | EFNB4 + LSDA     | EfficientNet  | <u>0.842</u> | <u>0.825</u> | <u>0.875</u> | <u>0.738</u> | <u>0.805</u> | <u>0.817</u> |
| **SVD (Ours)** | U-Net VAE   | U-Net VAE     | **0.869** <span style="color:blue;">(+0.027)</span> | **0.881** <span style="color:blue;">(+0.056)</span> | **0.894** <span style="color:blue;">(+0.019)</span> | **0.828** <span style="color:blue;">(+0.090)</span> | **0.899** <span style="color:blue;">(+0.094)</span> | **0.874** <span style="color:blue;">(+0.057)</span> |

### ✅ Model Trained on **DFD**

Cross-dataset evaluations using the **frame-level ROC-AUC** metric. All detectors are trained on **DFD** and evaluated on other datasets.

| Method    | Detector         | Backbone      | CDF-v2       | FF++         | CDF-v1       | DFDC         | DFDCP        | Avg.         |
|-----------|------------------|---------------|--------------|--------------|--------------|--------------|--------------|--------------|
| Naive     | Meso4            | MesoNet       | 0.708        | 0.660        | 0.621        | 0.544        | 0.596        | 0.626        |
| Naive     | MesoIncep        | MesoNet       | 0.733        | 0.679        | 0.639        | 0.580        | 0.660        | 0.658        |
| Naive     | CNN-Aug          | ResNet        | 0.741        | 0.695        | 0.664        | 0.610        | 0.630        | 0.668        |
| Naive     | Xception         | Xception      | 0.779        | 0.743        | 0.755        | 0.695        | 0.718        | 0.738        |
| Naive     | EfficientB4      | EfficientNet  | 0.792        | 0.755        | 0.769        | 0.710        | 0.736        | 0.752        |
| Spatial   | CapsuleNet       | Capsule       | 0.768        | 0.727        | 0.684        | 0.660        | 0.671        | 0.702        |
| Spatial   | FWA              | Xception      | 0.739        | 0.705        | 0.672        | 0.648        | 0.689        | 0.691        |
| Spatial   | Face X-ray       | HRNet         | 0.726        | 0.680        | 0.688        | 0.640        | 0.703        | 0.687        |
| Spatial   | FFD              | Xception      | 0.775        | 0.740        | 0.755        | 0.720        | 0.750        | 0.748        |
| Spatial   | CORE             | Xception      | 0.782        | 0.749        | 0.758        | 0.738        | 0.765        | 0.758        |
| Spatial   | Recce            | Custom        | 0.767        | -            | 0.739        | 0.708        | 0.727        | 0.735        |
| Spatial   | UCF              | Xception      | 0.782        | -            | 0.745        | 0.756        | 0.760        | 0.761        |
| Frequency | F3Net            | Xception      | 0.751        | 0.725        | 0.706        | 0.685        | 0.730        | 0.719        |
| Frequency | SPSL             | Xception      | 0.784        | 0.732        | 0.768        | 0.740        | 0.765        | 0.758        |
| Frequency | SRM              | Xception      | 0.772        | 0.754        | 0.782        | 0.718        | 0.748        | 0.755        |
| Frequency | EFNB4 + LSDA     | EfficientNet  | <u>0.836</u> | <u>0.820</u> | <u>0.830</u> | <u>0.772</u> | <u>0.798</u> | <u>0.811</u> |
| **SVD (Ours)** | U-Net VAE   | U-Net VAE     | **0.861** <span style="color:blue;">(+0.025)</span> | **0.873** <span style="color:blue;">(+0.053)</span> | **0.867** <span style="color:blue;">(+0.037)</span> | **0.832** <span style="color:blue;">(+0.060)</span> | **0.881** <span style="color:blue;">(+0.083)</span> | **0.863** <span style="color:blue;">(+0.052)</span> |

> Best results are in **bold**, second-best are <u>underlined</u>, and the improvements in ROC-AUC are shown in <span style="color:blue;">blue</span>.




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

