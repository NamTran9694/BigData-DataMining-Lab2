## MSCS_634_Lab_2 – KNN and Radius Neighbors Classifiers

**Course:** Advanced Big Data and Data Mining (MSCS-634-M20)
**Student:** Hoai Nam Tran  
**Lab:** Lab 2 – Classification Using KNN and RNN Algorithms

---

## Purpose

This lab explores the performance of two distance-based classifiers — **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)** — applied to the Wine Dataset from sklearn. The Wine Dataset contains 178 samples across 3 wine classes, described by 13 chemical features (e.g., alcohol content, color intensity, flavanoids).

The goal is to:
- Understand how parameter choices (k for KNN, radius for RNN) affect classification accuracy
- Visualize accuracy trends across different parameter values
- Compare the strengths and limitations of each model

---

## Key Insights

### KNN (k = 1, 5, 11, 15, 21)

| k | Test Accuracy |
|---|--------------|
| 1 | 0.778 |
| 5 | 0.722 |
| 11 | 0.750 |
| 15 | 0.750 |
| 21 | **0.778** |

- **Best performance: k=1 and k=21**, both achieving 77.8% accuracy.
- Accuracy showed a U-shaped trend — dipping at k=5 (lowest at 72.2%) before recovering at larger k values. This suggests the Wine Dataset has both local class structure (favoring small k) and enough global separation (favoring larger k).
- k=1 performed surprisingly well because the Wine classes are fairly well separated in feature space, meaning the single nearest neighbor is usually correct.

### RNN (radius = 350, 400, 450, 500, 550, 600)

| Radius | Test Accuracy |
|--------|--------------|
| 350 | **0.750** |
| 400 | 0.722 |
| 450 | 0.722 |
| 500 | 0.722 |
| 550 | 0.722 |
| 600 | 0.722 |

- **Best performance: radius=350** at 75.0%.
- Accuracy flatlined from radius 400 onward at 72.2%, indicating that larger radii pull in too many neighbors and dilute class-specific information.
- The flat trend suggests all radii ≥ 400 are capturing a similar large neighborhood, making the classifier increasingly coarse.

### KNN vs RNN Comparison
- **KNN outperformed RNN overall** — peak accuracy 77.8% vs 75.0%.
- KNN was more stable across parameter choices, while RNN quickly plateaued after radius 350.
- The overall accuracy range of 72–78% for both models is likely due to unscaled features. The Wine Dataset features vary dramatically in magnitude (e.g., alcohol ~11–14 vs. proline ~300–1700), which distorts distance calculations in favor of high-magnitude features. Applying `StandardScaler` before fitting would likely push both models well above 90%.
---

## Challenges and Decisions
- **Feature scaling:** The Wine Dataset features vary widely in scale (e.g., alcohol ~11–14 vs. proline ~278–1680). Distance-based methods like KNN and RNN are sensitive to this. For this lab, raw features were used as instructed, but applying `StandardScaler` before fitting would likely improve both models' accuracy significantly.
- **Radius value selection:** Choosing meaningful radius values required knowing the approximate scale of distances in the feature space. Values were selected (350–600) to ensure most test points had at least some neighbors while still observing variation in accuracy.
