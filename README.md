# Neural Credit Risk Explainer

This is a deep learning project for the Advanced Topics in Machine Learning course at the University of Rhode Island. The objective is to build a credit risk classification model using a feedforward neural network trained on tabular financial data. The project also explores model interpretability using SHAP and, time permitting, natural-language rationales via a small language model.

## Overview

This project investigates the relationship between several predictor variables—such as interest rate, annual income, and delayed payments—and a customer's credit status. The dataset used is the [Credit Score Classification dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) from Kaggle. Multiple deep learning models are implemented and compared to identify the most effective architecture for predicting credit risk. As a stretch goal, SHAP will be used to interpret model predictions, and a small LLM may be incorporated to assist in generating human-readable explanations.

## Background

Credit score classification and credit risk modeling are widely studied in both data science and finance. Traditional models like logistic regression and decision trees have long been used for credit risk assessment, but recent advances in machine learning have introduced more complex approaches with improved performance. Studies have shown that models such as Random Forest and XGBoost perform well on structured financial data. Deep neural networks have also demonstrated strong results. For example, a 2024 study on personal loan data found that “the deep neural network outperforms the other techniques in terms of all forecasting performance metrics” (Sami Mestiri, 2024). This project focuses on deep learning models and explores architectural variations to optimize performance.

## Dataset

- **Source**: [Credit Score Classification – Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- **Files**: `data/train.csv`, `data/test.csv`
- **Target Variable**: `Credit_Score` (mapped to binary: Poor = 1, Standard/Good = 0)

The dataset contains 27 predictors, including annual income, occupation, number of loans, outstanding debt, and investment amount. Twenty predictors are categorical, while the rest are numeric. Some columns are unique identifiers and are dropped during preprocessing. The dataset also contains missing values and outliers that must be addressed.

## Methods and Models

### Preprocessing

The raw dataset undergoes a preprocessing phase to prepare it for deep learning. Irrelevant and high-cardinality columns are removed, and data entry errors (e.g., negative age) are corrected. A Scikit-Learn `ColumnTransformer` pipeline is used to apply tailored transformations:

- **Numeric features**: imputed with median values and scaled using `StandardScaler`
- **Categorical features**: imputed with the most frequent value and one-hot encoded

The dataset is split into training and validation sets before fitting the pipeline to prevent data leakage. The result is a clean, fully numeric, and standardized dataset suitable for modeling.

### Modeling Strategy

The modeling approach involves iterative development and comparison of multiple deep learning architectures. Each model begins with a baseline version to establish performance benchmarks. These models are trained using appropriate loss functions and optimizers for binary classification, with callbacks to manage training (e.g., early stopping and checkpointing). After baseline evaluation, models are fine-tuned by adjusting architecture and hyperparameters. Performance is visualized and compared to identify the most effective approach.

### Models Explored

- **Feedforward Neural Network (Keras)**  
  A dense neural network with regularization and a sigmoid output layer. Architecture is tuned for optimal performance.

- **Transformer-Based Model (TensorFlow)**  
  A transformer architecture tested for its ability to capture complex patterns beyond traditional MLPs.

- **Autoencoder + Classifier Pipeline**  
  A two-step model where an autoencoder learns feature representations, and a classifier uses the encoded output to predict credit risk.

## SHAP Explainability

SHAP (SHapley Additive exPlanations) will be used to interpret model predictions. It assigns each feature a contribution value for individual predictions, helping to explain why the model classified a customer as high or low risk. SHAP plots will be used to visualize global and local feature importance.

## Conclusion

This project leverages deep learning to explore credit risk classification using a real-world financial dataset. By comparing multiple neural network architectures—including feedforward models, transformers, and autoencoder pipelines—we aim to understand how architectural choices affect predictive performance. Evaluation will focus on metrics such as AUC, precision, and recall. Beyond performance, the project emphasizes interpretability through SHAP and, if time permits, natural-language explanations via a small LLM. The goal is to uncover insights into how deep learning can be adapted for structured financial data and where it may outperform traditional approaches.
