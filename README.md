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

## Files

| File | Description |
|------|-------------|
| `knn_rnn_lab.ipynb` | Full Jupyter Notebook with code, plots, and analysis |
| `README.md` | This file |

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/MSCS_634_Lab_2.git
   cd MSCS_634_Lab_2
   ```

2. Install dependencies:
   ```bash
   pip install scikit-learn matplotlib pandas numpy jupyter
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook knn_rnn_lab.ipynb
   ```

4. Run all cells from top to bottom (Kernel → Restart & Run All).

---

## Key Insights

### KNN (k = 1, 5, 11, 15, 21)
- **k=1** is highly sensitive to noise — it classifies each point based on a single neighbor, which often leads to overfitting.
- Accuracy generally stabilized as k increased, reflecting the bias-variance tradeoff: larger k = smoother decision boundary but less sensitivity to local structure.
- Mid-range k values (around 5–11) tended to perform best on this dataset.

### RNN (radius = 350, 400, 450, 500, 550, 600)
- Smaller radii sometimes left test points with **no neighbors** (outliers), which reduced effective coverage and accuracy.
- Larger radii pulled in too many neighbors, essentially averaging across classes and reducing precision.
- A radius in the mid-range (around 450–500) balanced coverage and accuracy best.

### KNN vs RNN Comparison
- **KNN** was more consistent — it always classified every test point since it always finds exactly k neighbors.
- **RNN** was more sensitive to data density — sparse regions produced outlier points that couldn't be classified.
- For this dataset, KNN slightly outperformed RNN in reliability, though both models reached similar peak accuracy with well-chosen parameters.

---

## Challenges and Decisions

- **Outlier handling in RNN:** When a test point falls outside the specified radius, sklearn raises an error by default. This was handled by setting `outlier_label=-1` and excluding those points from the accuracy calculation, which is a fair approach since the model effectively abstains on those samples.
- **Feature scaling:** The Wine Dataset features vary widely in scale (e.g., alcohol ~11–14 vs. proline ~278–1680). Distance-based methods like KNN and RNN are sensitive to this. For this lab, raw features were used as instructed, but applying `StandardScaler` before fitting would likely improve both models' accuracy significantly.
- **Radius value selection:** Choosing meaningful radius values required knowing the approximate scale of distances in the feature space. Values were selected (350–600) to ensure most test points had at least some neighbors while still observing variation in accuracy.
