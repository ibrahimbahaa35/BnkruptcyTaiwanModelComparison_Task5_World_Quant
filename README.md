# Bankruptcy Prediction In Taiwan

## Overview
This project focuses on the prediction of corporate debt to classify whether an entity is bankrupt or not[cite: 1]. The analysis implements and compares three machine learning classification models:
* Logistic Regression[cite: 1]
* Random Forest Classifier[cite: 1]
* Gradient Boosting Classifier[cite: 1]

## Project Workflow
The pipeline is divided into three primary tasks:
1. **Data Extraction and Transformation:** Loading compressed JSON data from gzip files and extracting relevant observation records[cite: 1].
2. **Data Splitting:** Dividing the extracted data into training and testing sets[cite: 1].
3. **Model Training and Evaluation:** Training the classification models and comparing their performance metrics[cite: 1].

## Dependencies
The notebook imports several libraries for data processing and machine learning:
* **Data Manipulation & Analysis:** `pandas`, `numpy`[cite: 1]
* **Data Visualization:** `matplotlib.pyplot`[cite: 1]
* **Machine Learning (`scikit-learn`):** Used for model building (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`), preprocessing (`SimpleImputer`, `MinMaxScaler`, `Pipeline`), validation (`GridSearchCV`, `train_test_split`), and evaluation metrics (`accuracy_score`, `roc_auc_score`, `classification_report`, etc.)[cite: 1]
* **Imbalanced Data Handling:** `imblearn.over_sampling.RandomOverSampler`[cite: 1]
* **Standard Python Libraries:** `json`, `gzip`, `pathlib`, `pickle`[cite: 1]

## Dataset
The project relies on two compressed JSON files located in the `data_required` directory:
* `taiwan-bankruptcy-data.json.gz`[cite: 1]
* `taiwan-bankruptcy-data-test-features.json.gz`[cite: 1]

The loaded JSON payload structure contains `schema`, `metadata`, and `observations` (or `data`) keys[cite: 1]. The transformed records consist of observation dictionaries, each featuring an `id` and 95 distinct continuous numerical features (`feat_1` through `feat_95`)[cite: 1].
