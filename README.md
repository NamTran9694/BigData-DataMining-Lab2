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

### Effect of StandardScaler
`StandardScaler` was applied after the train/test split to transform all features to mean=0 and std=1. This is critical for distance-based models — without scaling, high-magnitude features like proline (range ~300–1700) dominate distance calculations over low-magnitude features like alcohol (range ~11–14). Scaling brought both models' accuracy from the 72–78% range up to 97–100%.

> The scaler was fit **only on the training set** and then applied to the test set to prevent data leakage.

### KNN (k = 1, 5, 11, 15, 21)

| k | Test Accuracy |
|---|--------------|
| 1 | 0.9444 |
| 5 | 0.9444 |
| 11 | 0.9444 |
| 15 | **0.9722** |
| 21 | 0.9444 |

- **Best performance: k=15** at 97.2% accuracy.
- After scaling, larger k values performed better — the scaled feature space is clean enough that averaging over more neighbors smooths out noise without losing class distinction.
- Scaling transformed accuracy from a peak of 77.8% (unscaled) to 97.2%.

### RNN (radius = 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)

| Radius | Test Accuracy |
|--------|--------------|
| 1.0 | 0.0000 |
| 1.5 | **1.0000** |
| 2.0 | 1.0000 |
| 2.5 | 0.9714 |
| 3.0 | 0.9444 |
| 3.5 | 0.9722 |

- **Best performance: radius=1.5** achieving a perfect **100% accuracy**.
- After scaling, radius values were adjusted to the 1.0–3.5 range (since scaled features now lie in ~-3 to +3), which revealed much more meaningful variation across radius values.
- Scaling transformed accuracy from a plateau of 72.2% (unscaled) to 100%.

### KNN vs RNN Comparison
- **RNN outperformed KNN** after scaling — 100% vs 97.2% at their best parameter values.
- Both models improved dramatically with scaling, confirming that unscaled feature magnitudes were the primary bottleneck.
- KNN remained more consistent across all parameter choices, while RNN was more sensitive but hit a higher peak at the right radius.

---

## Challenges and Decisions

- **Feature scaling:** The Wine Dataset features vary dramatically in magnitude (e.g., alcohol ~11–14 vs. proline ~300–1700). Without scaling, high-magnitude features dominated Euclidean distance calculations. Applying `StandardScaler` after the train/test split resolved this and improved both models significantly.
- **Preventing data leakage:** The scaler was fit exclusively on the training set using `fit_transform()`, then applied to the test set using `transform()` only. Fitting on the full dataset before splitting would leak test information into the model and produce artificially inflated accuracy.
- **Radius value adjustment:** After scaling, the original radius values (350–600) were no longer meaningful since scaled features are bounded to roughly ±3. Radius values were adjusted to 1.0–3.5 to reflect the new feature space scale.
- **Outlier handling in RNN:** `outlier_label=-1` was used to handle test points that fall outside the radius. Those points were excluded from the accuracy calculation since the model effectively abstains on them.

