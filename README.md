# Flight Delay Prediction

This Machine Learning project is dedicated to predicting flight departure delays using historical airline, airport, weather, and operational data. The task is formulated as a binary classification problem: predicting whether a flight departure will be delayed by 15 minutes or more.

The project explores the full machine learning workflow, from data import and data quality checks to feature engineering, model training, model evaluation, and interpretation of tree-based models.

## Objectives

- Build a machine learning model to predict whether a flight departure will be delayed by 15 minutes or more.
- Analyze how different factors, such as departure time, weather conditions, airport characteristics, airline activity, and aircraft schedule sequence, are related to flight delays.
- Compare several machine learning approaches, including KNN, Gaussian Naive Bayes, rule-based models, Decision Tree-based models, Random Forest, and LightGBM.
- Evaluate models using classification metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- Interpret tree-based models using feature importance and SHAP values.

## Dataset

The dataset for the project was taken from the Kaggle platform. The notebook contains the code used to download the data through `kagglehub`.

The project uses one CSV file:

- `full_data_flightdelay.csv`

This file contains both the target variable and explanatory features. The target variable is `DEP_DEL15`, which indicates whether a flight departure was delayed by 15 minutes or more. The dataset also includes features such as `DEP_HOUR`, `DEPARTING_AIRPORT`, `CARRIER_NAME`, `PRCP`, `SNOW`, `AWND`, `CONCURRENT_FLIGHTS`, and `SEGMENT_NUMBER`.

## Project Workflow

The implementation is organized into several main stages:

- **Data Import**: downloads and loads the Kaggle dataset.
- **Data Understanding**: explores the target distribution, categorical feature distributions, delay rates, and numerical feature distributions.
- **Data Quality Analysis**: checks missing values, duplicates, data types, suspicious values, weather plausibility, and logical consistency.
- **Data Preparation**: handles feature transformations, outlier detection with Isolation Forest, outlier flag creation, cardinality checks, correlation analysis, mutual information, feature binning, and removal of redundant features.
- **Model-Specific Preprocessing and Modeling**: builds preprocessing pipelines adapted to different model types, including scaling for KNN and encoding strategies for categorical variables.
- **Model Evaluation**: compares models using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and ROC curves.
- **Model Interpretation**: analyzes feature importance and SHAP values for tree-based models.

## Models

The following models were tested:

- K-Nearest Neighbors
- K-Nearest Neighbors with PCA
- RIPPER rule-based classifier
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- LightGBM
- Feature-engineered LightGBM with hyperparameter tuning

The best overall performance was achieved by the feature-engineered LightGBM model with hyperparameter tuning, which provided the strongest balance between precision and recall.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── flight_delay_prediction.ipynb
├── Figures
│   ├── Boxplot_diagram1.png
│   ├── Boxplot_diagram2.png
│   ├── Feature importance random forest.png
│   ├── Feature impotance final Light GBM.png
│   ├── Flight Delay Distribution.png
│   ├── Model results summary.png
│   ├── Roc curve Gaussian.png
│   ├── Roc curve decision tree.png
│   ├── Roc curve final LightGBM.png
│   ├── Roc curve knn.png
│   ├── Roc curve random forest.png
│   ├── SHAP values final Light GBM.png
│   └── SHAP values random forest.png
├── notes
│   └── column_description.md
└── Report
    ├── Flight_Delay_Prediction_Report.pdf
    └── Flight_Delay_Prediction_Presentation.pdf
```

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open and run the notebook:

```text
flight_delay_prediction.ipynb
```

The dataset is downloaded inside the notebook using `kagglehub`, so Kaggle access may be required.

## Results Summary

The results show that flight delay prediction is a challenging classification task. The available features capture meaningful patterns, especially those related to departure hour, precipitation, departure airport, and aircraft schedule sequence, but they do not fully explain flight delays.

| Model | Dataset | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM - Feature Engineered + Hyperparameters | full (1.3M) | 0.67 | 0.32 | 0.65 | 0.43 | 0.73 |
| LightGBM | full (1.3M) | 0.64 | 0.29 | 0.65 | 0.40 | 0.69 |
| Random Forest | full (1.3M) | 0.58 | 0.27 | 0.69 | 0.39 | 0.67 |
| Decision Tree (full) | full (1.3M) | 0.60 | 0.27 | 0.65 | 0.38 | 0.67 |
| Decision Tree (sample) | sample (20k) | 0.59 | 0.26 | 0.64 | 0.37 | 0.66 |
| GaussianNB | full (1.3M) | 0.71 | 0.28 | 0.33 | 0.30 | 0.61 |
| GaussianNB | sample (20k) | 0.72 | 0.27 | 0.30 | 0.29 | 0.61 |
| KNN | sample (20k) | 0.79 | 0.32 | 0.12 | 0.17 | 0.59 |
| KNN + PCA | sample (20k) | 0.79 | 0.32 | 0.12 | 0.17 | 0.59 |
| RIPPER | sample (20k) | 0.81 | 0.33 | 0.00 | 0.00 | 0.50 |

Simple baseline models were limited by the scale of the dataset and weak individual feature-target relationships: distance-based and probabilistic approaches struggled to capture the complex interactions between features. Tree-based models performed better because they can model non-linear relationships and feature interactions without requiring scaling or strong distributional assumptions. However, the overall predictive performance remained moderate, suggesting that richer real-time operational data would be needed for more accurate delay prediction.

## Future Work

Several improvements could be explored in future work:

- Add richer real-time operational data, such as live weather updates, aircraft rotation status, crew availability, gate availability, air traffic control restrictions, and previous flight delay information.
- Use time-based validation instead of only random train-test splitting to better simulate real-world prediction on future flights.
- Tune classification thresholds to improve the balance between precision and recall, especially because delayed flights are the minority class.
- Test more advanced gradient boosting configurations and perform broader hyperparameter optimization.
- Analyze model performance separately by airport, carrier, month, and departure time to identify where the model works well and where it fails.
- Calibrate predicted probabilities so that delay risk scores are easier to interpret and use in operational decision-making.