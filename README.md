# Satellite Imagery-Based Property Valuation

## Overview
This project builds a **multimodal regression pipeline** to predict residential property prices by combining:
- **Tabular housing attributes** (size, rooms, location, quality)
- **Satellite imagery** capturing neighborhood and environmental context

The goal is to evaluate whether visual cues such as greenery, road layout, and neighborhood density can complement traditional real-estate features and improve valuation accuracy.

---

## Problem Statement
Traditional house price models rely heavily on structured features (e.g., square footage, bedrooms). However, intangible factors such as **curb appeal**, **neighborhood quality**, and **environmental context** are difficult to encode numerically.

This project explores whether **satellite imagery**, processed via deep learning, can capture these factors and improve price prediction.

---

## Dataset
- **Base Dataset:** King County House Sales (Kaggle)
- **Records:** ~16,000 properties
- **Target:** `price`
- **Key Tabular Features:**  
  `bedrooms`, `bathrooms`, `sqft_living`, `grade`, `lat`, `long`, etc.

### Visual Data
- Satellite images were programmatically fetched using **Mapbox Static Images API**
- Each property image is centered on its latitude/longitude

---

## Project Structure
data/
├── raw/ # Original CSV files
├── images/ # Downloaded satellite images
└── processed/ # Image embeddings

notebooks/
├── 01_eda_tabular.ipynb
├── 02_image_exploration.ipynb
├── 04_multimodal_regression.ipynb
├── 05_neural_multimodal.ipynb
└── 06_gradcam_explainability.ipynb

src/
├── data_fetcher.py # Image download pipeline
├── dataset.py # Dataset utilities
├── models.py # ML & neural models
└── explainability.py # Grad-CAM helpers

---

## Methodology

### 1. Tabular Baseline
- Log-transformed price
- Standardized numeric features
- Random Forest Regression

### 2. Image Feature Extraction
- Pretrained **ResNet18**
- Extracted 512-dimensional embeddings per image
- Saved embeddings for reuse

### 3. Multimodal Fusion Strategies
- **Late Fusion + Random Forest**
- **PCA-reduced image embeddings**
- **Neural Multimodal MLP**

### 4. Explainability
- **Grad-CAM** applied to CNN
- Visualized spatial regions influencing model attention

---

## Results

| Model | RMSE (log-price) | R² |
|-----|-----------------|----|
| Tabular Random Forest | **0.179** | **0.884** |
| Multimodal RF (PCA) | 0.190 | 0.869 |
| Neural Multimodal MLP | 0.251 | 0.772 |

**Key Insight:**  
Tabular features dominate predictive performance, but satellite imagery captures meaningful qualitative neighborhood signals.

---

## Explainability (Grad-CAM)
Grad-CAM visualizations show the CNN primarily focuses on:
- Green cover / vegetation
- Neighborhood layout
- Residential density

This suggests satellite imagery captures **environmental and aesthetic cues**, even if they do not significantly reduce RMSE.

---

## Conclusion
- Strong tabular baselines are difficult to outperform
- Satellite imagery adds **interpretability and spatial insight**
- Multimodal learning is valuable for **explainable real estate analytics**

---

## How to Run
1. Set Mapbox token:
```bash
export MAPBOX_TOKEN=your_token_here
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run notebooks in order inside notebooks/

## Repository Notes
- `dataset.py`, `models.py`, and `explainability.py` contain modular helper code used by the notebooks.
- All required deliverables are accessible via the notebooks listed above.
- `preprocessing.ipynb` and `model_training.ipynb` are the primary notebooks required by the problem statement.
- Additional notebooks provide deeper analysis, neural multimodal experiments, and Grad-CAM explainability.
- The `src/` directory contains modular helper code used by the notebooks.

Note: GPU is optional. All code runs on CPU by default.
