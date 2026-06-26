# Health Insurance Risk Classification Using Machine Learning

This repository contains R code and an anonymized sample dataset for machine learning-based risk classification in private health insurance.

The project was developed as part of a PhD research study in insurance analytics and predictive modelling.

## Dataset

The dataset contains 500 anonymized policy-level observations derived from private health insurance claim records. Each row represents a pseudo-insured policy unit, not an original claim-level record.

For confidentiality reasons, the original dataset is not publicly available. The shared dataset includes only processed modelling variables such as claim frequency, total claim amount, reporting and payment delays, encoded service/risk categories, age, gender, COVID-19 indicator, and the target class.

The target variable `TargetClass` was created using K-means clustering on the log-transformed total claim amount and includes three classes: `Low`, `Medium`, and `High`.

## Methods

The R code applies several machine learning models, including Random Forest, Decision Tree, XGBoost, SVM and LASSO Multinomial Regression.

Model performance is evaluated using accuracy, balanced accuracy, precision, recall, F1-score, confusion matrix, Type I and Type II errors, and AUC.

## Confidentiality

The original private health insurance dataset is not publicly available due to confidentiality restrictions.
This repository is provided for academic transparency as part of a PhD research project. The shared dataset is an anonymized sample and the code is intended to document the modelling workflow used in the dissertation. 
The dataset does not contain names, contract numbers, policy numbers, dates of birth, exact claim dates, or real personal identifiers.

## Author

Esmeralda Brati

## License

The R code is provided under the MIT License. The anonymized sample dataset is provided for academic demonstration and reproducibility purposes only.
