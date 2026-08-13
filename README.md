# 📱 Smartphone Addiction Classification

This repository contains a Machine Learning solution for the [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e8). The goal is to predict smartphone addiction based on user activity, screen time, and demographic features.

🔗 **Links:**
- [Kaggle Notebook (EN)](https://www.kaggle.com/code/nikitasem/smartphone-addiction-classification)
- [Google Colab Notebook (RU)](https://colab.research.google.com/drive/1eW5JU744dGfKM2g6Zucr7KEq50kwmS-O?usp=sharing)

---

## What's done

- **Exploratory Data Analysis (EDA):** Detailed analysis of feature distributions, target relationship, and categorical proportions.
- **Feature Engineering:** Derived relative activity metrics (e.g., social/gaming/work ratios to total screen time).
- **Data Preprocessing:** Creating a data pipeline for the complete data processing cycle and feeding the data into the models.
- **Models:** Trained `LogisticRegression`, `RandomForest`, `CatBoost`, and a custom `PyTorch Fully Connected Neural Network`.
- **Hyperparameter Tuning:** Automated search using `Optuna`.
- **Experiment Tracking:** Full metric, parameter, and artifact logging with `MLflow`.
- **Ensembling:** Applied Rank Averaging blending on best-performing models to stabilize test predictions.

---

## 📊 Validation ROC-AUC

| Model | ROC-AUC |
| :--- | :---: |
| **CatBoost (v2)** | **0.963** |
| **CatBoost (v1)** | 0.962 |
| **RandomForest** | 0.942 |
| **FullyConnected NN** | 0.933 |
| **Logistic Regression** | 0.918 |

## 📊 Test ROC-AUC

| Model | ROC-AUC |
| :--- | :---: |
| **CatBoost (v2)** | 0.96365 |
| **CatBoost (v1)** | 0.96314 |
| **Weight Blending (2 CatBoost)** | **0.96449** |
| **Weight Blending (All Models)** | 0.96037 |

---

## 📈 MLflow Experiment Tracking

![MLflow Metrics](https://github.com/user-attachments/assets/612553c5-3b12-427f-870c-2762eed52b39)
![MLflow Artifacts](https://github.com/user-attachments/assets/852f6d51-387e-4fb6-a91e-0828fbcf2cf1)
