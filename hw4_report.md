# Project Report: Classification Methods for Imbalanced Data

## Introduction
This project focuses on addressing the challenges of classification when the dataset has a significant class imbalance. Specifically, we compare the effectiveness of three oversampling techniques: **Synthetic Minority Oversampling Technique (SMOTE)**, **ADASYN (Adaptive Synthetic Sampling)**, and **Normalizing Flows**. The performance of these methods is evaluated using the Porto Seguro Safe Driver Prediction dataset from Kaggle.

## Methodology
### Data Preparation
- **Dataset**: The Porto Seguro Safe Driver Prediction dataset, pre-split into training and test sets.
- **Sampling Strategy**: We subset 10% of observations from each class in the training data to make the computations manageable.
- **Oversampling Techniques**:
  - **SMOTE**: Generates synthetic samples for the minority class by interpolating between existing samples.
  - **ADASYN**: Generates synthetic samples for the minority class, with a focus on more difficult samples.
  - **Normalizing Flows**: A generative modeling technique used to sample from complex distributions, enabling oversampling of the minority class.

### Evaluation Metrics
- **Stratified K-Fold Cross-Validation**: We use a 5-fold stratified cross-validation approach to ensure an equal distribution of classes in each fold.
- **Metrics**: 
  - **Accuracy**
  - **Recall**
  - **Precision**
  - **F1-score**
  - **Confusion Matrices**: To visualize the performance in terms of true positives, true negatives, false positives, and false negatives.

## Results
The results of each oversampling technique are presented below, including the confusion matrices for a detailed performance breakdown.

### 1. SMOTE
**Mean Metrics**:
- **Accuracy**: _X.XXXX_ ± _Y.YYYY_
- **Recall**: _X.XXXX_ ± _Y.YYYY_
- **Precision**: _X.XXXX_ ± _Y.YYYY_
- **F1-score**: _X.XXXX_ ± _Y.YYYY_

**Confusion Matrices**:
![Confusion Matrix for SMOTE Fold 1](path_to_image_1)
![Confusion Matrix for SMOTE Fold 2](path_to_image_2)
...

### 2. ADASYN
**Mean Metrics**:
- **Accuracy**: _X.XXXX_ ± _Y.YYYY_
- **Recall**: _X.XXXX_ ± _Y.YYYY_
- **Precision**: _X.XXXX_ ± _Y.YYYY_
- **F1-score**: _X.XXXX_ ± _Y.YYYY_

**Confusion Matrices**:
![Confusion Matrix for ADASYN Fold 1](path_to_image_3)
![Confusion Matrix for ADASYN Fold 2](path_to_image_4)
...

### 3. Normalizing Flows
**Mean Metrics**:
- **Accuracy**: _X.XXXX_ ± _Y.YYYY_
- **Recall**: _X.XXXX_ ± _Y.YYYY_
- **Precision**: _X.XXXX_ ± _Y.YYYY_
- **F1-score**: _X.XXXX_ ± _Y.YYYY_

**Confusion Matrices**:
![Confusion Matrix for Normalizing Flows Fold 1](path_to_image_5)
![Confusion Matrix for Normalizing Flows Fold 2](path_to_image_6)
...

## Test Set Predictions
- The models were trained on the full training data with oversampling and used to generate predictions for the test set. The predictions were saved in the following CSV files:
  - [SMOTE Predictions](./mnt/data/SMOTE_predictions.csv)
  - [ADASYN Predictions](./mnt/data/ADASYN_predictions.csv)
  - [Normalizing Flows Predictions](./mnt/data/Normalizing%20Flows_predictions.csv)

## Conclusion
- **Performance Comparison**: The performance of each technique varied based on the metric considered. Recall and F1-score were particularly crucial for evaluating the model's ability to correctly identify the minority class.
- **Insights**: Normalizing Flows provided a unique approach to oversampling, potentially capturing more complex distributions of the minority class. However, SMOTE and ADASYN were more straightforward to implement and interpret.

## Future Work
- **Model Tuning**: Further work could involve hyperparameter tuning for each technique to improve performance.
- **Other Models**: Exploring other classifiers, such as XGBoost or Neural Networks, to see how oversampling techniques influence their performance.
- **Feature Engineering**: Additional feature engineering might improve the overall model performance.
