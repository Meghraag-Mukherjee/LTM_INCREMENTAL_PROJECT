# RetailMind AI

RetailMind AI is an incremental retail intelligence project covering machine learning, deep learning, computer vision, explainable AI, and time-series forecasting.

The project evolves one increment at a time, with each stage extending the previous RetailMind codebase.

---

## Project Overview

RetailMind currently contains three major AI capabilities:

1. **Customer Churn Prediction**
   - Classical machine learning
   - Hyperparameter tuning
   - Feature engineering
   - Ensemble learning
   - Artificial Neural Networks

2. **Product Image Classification**
   - CNN from scratch
   - Transfer learning with MobileNetV2
   - Grad-CAM explainability

3. **Demand Forecasting**
   - LSTM next-step forecasting
   - Seq2Seq multi-horizon forecasting
   - Attention mechanism

---

## Project Structure

```text
RetailMind/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── fashion_products/
│   │   ├── Bags/
│   │   ├── Shirts/
│   │   ├── Shoes/
│   │   ├── Trousers/
│   │   └── Watches/
│   └── store_item_demand/
│       └── train.csv
├── models/
│   ├── churn_pipeline.joblib
│   ├── champion_model.joblib
│   ├── churn_v3.joblib
│   ├── churn_final.joblib
│   ├── churn_ann.keras
│   ├── ann_preprocessor.joblib
│   ├── fashion_mnist_cnn.keras
│   ├── catalog_cnn.keras
│   ├── catalog_transfer.keras
│   ├── catalog_class_names.json
│   ├── demand_lstm.keras
│   ├── demand_scaler.json
│   ├── demand_seq2seq_attention.keras
│   └── seq2seq_config.json
├── results/
│   ├── gradcam/
│   ├── demand_forecast.csv
│   ├── demand_metrics.csv
│   ├── demand_forecast.png
│   ├── seq2seq_forecast.csv
│   ├── seq2seq_metrics.csv
│   ├── horizon_metrics.csv
│   ├── attention_weights.csv
│   └── attention_heatmap.png
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   ├── compare_models.py
│   ├── tune_models.py
│   ├── features.py
│   ├── finalize_model.py
│   ├── train_ann.py
│   ├── predict_ann.py
│   ├── fashion_mnist_cnn.py
│   ├── train_catalog_cnn.py
│   ├── predict_catalog.py
│   ├── train_transfer.py
│   ├── gradcam.py
│   ├── explain_product.py
│   ├── train_demand_lstm.py
│   ├── predict_demand.py
│   ├── train_seq2seq_attention.py
│   └── predict_multi_horizon.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Phase 1: Customer Churn Prediction

The first phase builds an end-to-end churn prediction system using the Telco Customer Churn dataset.

```text
Churn
No  -> 0
Yes -> 1
```

The goal is to identify customers who are likely to churn so retention actions can be prioritized.

---

## Day 1: Baseline Churn Model

### Objective

Build the first end-to-end machine-learning pipeline for customer churn prediction.

### Workflow

```text
Telco CSV
    ↓
Data cleaning
    ↓
Train/test split
    ↓
Numerical and categorical preprocessing
    ↓
Logistic Regression
    ↓
Evaluation
    ↓
Persist model
```

### Preprocessing

- Convert `TotalCharges` to numeric.
- Remove `customerID`.
- Impute numerical values using the median.
- Standardize numerical features with `StandardScaler`.
- Impute categorical values with the most frequent category.
- Encode categorical features with `OneHotEncoder`.
- Use a stratified 80/20 train-test split.

### Model

```text
Logistic Regression
```

The preprocessing steps and classifier are stored in one scikit-learn pipeline.

### Results

| Metric | Value |
|---|---:|
| Accuracy | ACTUAL_VALUE |
| Precision | ACTUAL_VALUE |
| Recall | ACTUAL_VALUE |
| ROC-AUC | ACTUAL_VALUE |

### Saved model

```text
models/churn_pipeline.joblib
```

---

## Day 2: Classical Model Comparison

### Objective

Compare multiple algorithms using the same data split and preprocessing pipeline.

### Models

- Decision Tree
- Random Forest
- Support Vector Machine

The champion is selected with emphasis on churn recall.

### Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Decision Tree | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE |
| Random Forest | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE |
| SVM | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE |

### Saved champion

```text
models/champion_model.joblib
```

Comparison results:

```text
models/day2_model_comparison.csv
```

---

## Day 3: Hyperparameter Tuning

### Objective

Improve the candidate models using hyperparameter tuning and cross-validation.

### Techniques

- `GridSearchCV`
- Five-fold `StratifiedKFold`
- Training versus validation comparison
- Bias/variance diagnosis
- Learning curves

The main GridSearchCV selection metric is:

```python
scoring="recall"
```

### Bias and variance

```text
Low training + low validation
        ↓
Possible underfitting / high bias
```

```text
High training + much lower validation
        ↓
Possible overfitting / high variance
```

### Results

| Model | CV Train Recall | CV Validation Recall | Test Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Decision Tree | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE |
| Random Forest | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE |
| SVM | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE |

Best parameters:

```text
ACTUAL_BEST_PARAMETERS
```

### Saved tuned champion

```text
models/churn_v3.joblib
```

Additional outputs:

```text
models/day3_tuning_results.csv
models/day3_learning_curve.csv
```

---

## Day 4: Feature Engineering and Final Phase-1 Model

### Objective

Improve churn prediction using domain feature engineering, dimensionality reduction, and ensemble algorithms.

### Engineered features

- Tenure group
- Average charge per month
- Charge difference
- Month-to-month contract flag
- Electronic-check flag
- Fiber-optic flag
- Automatic-payment flag
- Number of optional services
- Security/support flag
- Senior customer without partner flag

### PCA

```python
PCA(n_components=0.95)
```

This retains enough principal components to explain approximately 95% of transformed feature variance.

### Models compared

- Day-3 tuned champion
- Random Forest with engineered features and PCA
- XGBoost with engineered features and PCA

### Results

| Model | CV Recall | Test Recall | Test ROC-AUC |
|---|---:|---:|---:|
| Day-3 Champion | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE |
| Random Forest + Features + PCA | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE |
| XGBoost + Features + PCA | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE |

### Final Phase-1 model

```text
Selected model: ACTUAL_MODEL_NAME
```

```text
models/churn_final.joblib
```

The saved artifact contains the transformations required by the selected model.

---

# Phase 2: Deep Learning and Computer Vision

## Day 5: ANN Churn Prediction

### Objective

Determine whether a neural network improves churn prediction compared with the Phase-1 classical champion.

### Architecture

```text
Customer features
      ↓
Dense(64, ReLU)
      ↓
Batch Normalization
      ↓
Dropout
      ↓
Dense(32, ReLU)
      ↓
Dropout
      ↓
Dense(16, ReLU)
      ↓
Dense(1, Sigmoid)
      ↓
Churn probability
```

### Training configuration

- Optimizer: Adam
- Learning rate: `0.001`
- Loss: Binary cross-entropy
- Maximum epochs: 100
- Batch size: 32
- Early stopping on validation loss
- Learning-rate reduction
- Balanced class weights

### ANN versus classical ML

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Phase-1 Classical Champion | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE |
| Keras ANN | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE | ACTUAL_VALUE |

### Saved files

```text
models/churn_ann.keras
models/ann_preprocessor.joblib
models/ann_comparison.csv
models/ann_training_history.csv
```

---

## Day 6: CNN Product Classification

### Objective

Build a CNN that automatically categorizes fashion-product images.

### Fashion-MNIST prototype

```text
28 x 28 grayscale image
        ↓
Conv2D + MaxPooling
        ↓
Conv2D + MaxPooling
        ↓
Conv2D + MaxPooling
        ↓
Dropout + Dense
        ↓
Softmax
        ↓
10 fashion classes
```

Fashion-MNIST test accuracy:

```text
ACTUAL_VALUE
```

Saved prototype:

```text
models/fashion_mnist_cnn.keras
```

### Real catalog CNN

The real classifier uses:

- Directory-based image loading
- Image resizing
- Pixel normalization
- Horizontal flipping
- Rotation
- Zoom
- Translation
- Convolution blocks
- Batch normalization
- Max pooling
- Dropout
- Softmax classification

Validation accuracy:

```text
ACTUAL_VALUE
```

Saved files:

```text
models/catalog_cnn.keras
models/catalog_class_names.json
results/catalog_training_history.csv
results/catalog_confusion_matrix.csv
```

---

## Day 7: Transfer Learning

### Objective

Improve the from-scratch product classifier using a pretrained MobileNetV2 backbone.

### Workflow

```text
Product image
       ↓
MobileNetV2 pretrained on ImageNet
       ↓
Frozen backbone
       ↓
Global Average Pooling
       ↓
Dropout
       ↓
Classification head
       ↓
Product category
```

The pretrained network is loaded with:

```python
MobileNetV2(
    weights="imagenet",
    include_top=False
)
```

The base is initially frozen:

```python
base_model.trainable = False
```

### Comparison

| Model | Validation Accuracy |
|---|---:|
| Day-6 CNN from scratch | ACTUAL_VALUE |
| Day-7 MobileNetV2 | ACTUAL_VALUE |
| Optional fine-tuned MobileNetV2 | ACTUAL_VALUE |

### Saved model

```text
models/catalog_transfer.keras
```

Object detection is not part of this increment.

---

## Day 8: Grad-CAM Explainability

### Objective

Make the transfer-learning classifier explainable by showing which image regions contributed to the selected prediction.

No new classifier is trained. The Day-7 model is reused.

### Workflow

```text
Product image
      ↓
Day-7 classifier
      ↓
Predicted class
      ↓
GradientTape
      ↓
Class gradients
      ↓
Convolutional feature maps
      ↓
Grad-CAM heatmap
      ↓
Overlay on original image
```

For each inspected image, the module returns:

- Predicted category
- Prediction confidence
- Grad-CAM heatmap
- Heatmap overlay

Example:

```python
result = explain_product("path/to/product.jpg")
```

Outputs:

```text
results/gradcam/
```

Grad-CAM visualizes model activation for a selected prediction. It should not be treated as proof of causal influence.

---

# Phase 3: Demand Forecasting

## Day 9: LSTM Demand Forecasting

### Objective

Forecast next-period product demand using historical sales sequences.

### Data preparation

- Parse dates
- Select one store-item series
- Sort chronologically
- Resample to daily frequency
- Fill missing dates
- Fit scaling values using training data only
- Hold out the latest observations for testing
- Create 30-day look-back windows

No random train-test split is used.

### Windowing

```text
Previous 30 days
       ↓
LSTM
       ↓
Next day's sales
```

Input shape:

```text
(samples, 30, 1)
```

### Architecture

```text
30-day sales sequence
       ↓
LSTM(64)
       ↓
Dropout
       ↓
LSTM(32)
       ↓
Dropout
       ↓
Dense(16, ReLU)
       ↓
Dense(1)
       ↓
Next-day demand
```

### Results

| Metric | Value |
|---|---:|
| MAE | ACTUAL_VALUE |
| RMSE | ACTUAL_VALUE |

### Saved outputs

```text
models/demand_lstm.keras
models/demand_scaler.json
results/demand_forecast.csv
results/demand_metrics.csv
results/demand_training_history.csv
results/demand_forecast.png
```

---

## Day 10: Seq2Seq + Attention Forecasting

### Objective

Upgrade the single-step LSTM into a multi-horizon demand forecaster.

### Configuration

```text
Historical look-back: 30 days
Forecast horizon: 7 days
```

Input shape:

```text
(samples, 30, 1)
```

Target shape:

```text
(samples, 7, 1)
```

### Architecture

```text
30 historical days
       ↓
Encoder LSTM
       ↓
Sequence outputs + hidden state + cell state
       ↓
Decoder LSTM
       ↓
Additive Attention over historical days
       ↓
TimeDistributed Dense layers
       ↓
7 future demand values
```

### Overall results

| Metric | Value |
|---|---:|
| Multi-horizon MAE | ACTUAL_VALUE |
| Multi-horizon RMSE | ACTUAL_VALUE |

### Per-horizon results

| Horizon | MAE | RMSE |
|---|---:|---:|
| Day +1 | ACTUAL_VALUE | ACTUAL_VALUE |
| Day +2 | ACTUAL_VALUE | ACTUAL_VALUE |
| Day +3 | ACTUAL_VALUE | ACTUAL_VALUE |
| Day +4 | ACTUAL_VALUE | ACTUAL_VALUE |
| Day +5 | ACTUAL_VALUE | ACTUAL_VALUE |
| Day +6 | ACTUAL_VALUE | ACTUAL_VALUE |
| Day +7 | ACTUAL_VALUE | ACTUAL_VALUE |

### Attention explainability

Attention weights are exported for each future step against the historical input positions.

```text
results/attention_weights.csv
results/attention_heatmap.png
```

Attention weights show internal model weighting and should not be interpreted as proof of causal importance.

### Saved outputs

```text
models/demand_seq2seq_attention.keras
models/seq2seq_config.json
results/seq2seq_forecast.csv
results/seq2seq_metrics.csv
results/horizon_metrics.csv
results/attention_weights.csv
results/attention_heatmap.png
results/seq2seq_training_history.csv
```

---

# RetailMind Evolution

```text
DAY 1
Logistic Regression baseline
       ↓
DAY 2
Decision Tree / Random Forest / SVM
       ↓
DAY 3
GridSearchCV + cross-validation
       ↓
DAY 4
Feature engineering + PCA + ensembles
       ↓
DAY 5
ANN churn classifier
       ↓
DAY 6
CNN product classifier
       ↓
DAY 7
MobileNetV2 transfer learning
       ↓
DAY 8
Grad-CAM explainability
       ↓
DAY 9
LSTM next-day demand forecasting
       ↓
DAY 10
Seq2Seq + Attention multi-horizon forecasting
```

---

# Current Capabilities

## Customer Intelligence

```text
Customer information
        ↓
Churn prediction
        ↓
Retention-risk signal
```

## Catalog Intelligence

```text
Product image
       ↓
CNN / MobileNetV2
       ↓
Product category
       ↓
Grad-CAM
       ↓
Visual explanation
```

## Inventory Intelligence

```text
Historical sales
       ↓
LSTM / Seq2Seq
       ↓
Future demand
       ↓
Inventory-planning signal
```

---

# Main Technologies

## Machine Learning

- pandas
- scikit-learn
- XGBoost
- joblib

## Deep Learning

- TensorFlow
- Keras

## Techniques

- Logistic Regression
- Decision Trees
- Random Forest
- SVM
- GridSearchCV
- Cross-validation
- PCA
- Feature engineering
- Ensemble learning
- Artificial Neural Networks
- CNN
- Transfer learning
- MobileNetV2
- Grad-CAM
- LSTM
- Seq2Seq
- Attention
- Time-series forecasting

---

# Installation

## Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Running the Project

## Day 1

```bash
python src/train.py
python src/predict.py
```

## Day 2

```bash
python src/compare_models.py
```

## Day 3

```bash
python src/tune_models.py
```

## Day 4

```bash
python -m src.finalize_model
```

## Day 5

```bash
python -m src.train_ann
python -m src.predict_ann
```

## Day 6

```bash
python -m src.fashion_mnist_cnn
python -m src.train_catalog_cnn
python -m src.predict_catalog "path/to/product.jpg"
```

## Day 7

```bash
python -m src.train_transfer
```

## Day 8

```bash
python -m src.explain_product "data/fashion_products/Shoes/shoe001.jpg"
```

## Day 9

```bash
python -m src.train_demand_lstm
python -m src.predict_demand
```

## Day 10

```bash
python -m src.train_seq2seq_attention
python -m src.predict_multi_horizon
```

---

# Model Artifacts

| Module | Model |
|---|---|
| Day-1 churn baseline | `models/churn_pipeline.joblib` |
| Day-2 champion | `models/champion_model.joblib` |
| Day-3 tuned champion | `models/churn_v3.joblib` |
| Phase-1 final churn model | `models/churn_final.joblib` |
| ANN churn model | `models/churn_ann.keras` |
| CNN catalog classifier | `models/catalog_cnn.keras` |
| Transfer-learning classifier | `models/catalog_transfer.keras` |
| LSTM demand forecaster | `models/demand_lstm.keras` |
| Seq2Seq attention forecaster | `models/demand_seq2seq_attention.keras` |

---

# Key Results Summary

Replace each placeholder with the actual result generated by the project.

| Increment | Model | Main Metric | Result |
|---|---|---|---:|
| Day 1 | Logistic Regression | Recall | ACTUAL_VALUE |
| Day 2 | DT/RF/SVM Champion | Recall | ACTUAL_VALUE |
| Day 3 | Tuned Champion | CV Recall | ACTUAL_VALUE |
| Day 4 | Final Classical Model | Recall | ACTUAL_VALUE |
| Day 5 | ANN | Recall | ACTUAL_VALUE |
| Day 6 | CNN | Validation Accuracy | ACTUAL_VALUE |
| Day 7 | MobileNetV2 | Validation Accuracy | ACTUAL_VALUE |
| Day 9 | LSTM | MAE | ACTUAL_VALUE |
| Day 10 | Seq2Seq + Attention | MAE | ACTUAL_VALUE |

---

# Experimental Principles

- Keep training and evaluation data separate.
- Use chronological splits for time-series data.
- Fit preprocessing and scaling only on training data.
- Compare classification models with more than accuracy alone.
- Emphasize churn recall because missing an actual churner may be costly.
- Use early stopping for neural-network training.
- Reuse saved preprocessing during inference.
- Record actual experimental outputs instead of assumed values.
- Retain complex models only when measured performance justifies the additional complexity.

---

# Conclusion

RetailMind has evolved from a customer-churn baseline into a multi-module retail AI system supporting:

- Customer churn prediction
- Classical and deep-learning benchmarking
- Product-image classification
- Transfer learning
- Visual explainability with Grad-CAM
- Single-step demand forecasting with LSTM
- Multi-horizon forecasting with Seq2Seq and attention

The project demonstrates how multiple AI techniques can be integrated incrementally while preserving reproducible training, evaluation, persistence, and inference workflows.
