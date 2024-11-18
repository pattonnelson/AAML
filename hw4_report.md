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
 ```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data_subset = train_data.groupby('target').apply(lambda x: x.sample(frac=0.1, random_state=42)).reset_index(drop=True)

X_train = train_data_subset.drop(['id', 'target'], axis=1)
y_train = train_data_subset['target']
X_test = test_data.drop('id', axis=1)
```

### Evaluation Metrics
- **Stratified K-Fold Cross-Validation**: We use a 5-fold stratified cross-validation approach to ensure an equal distribution of classes in each fold.
- **Metrics**: 
  - **Accuracy**
  - **Recall**
  - **Precision**
  - **F1-score**
  - **Confusion Matrices**: To visualize the performance in terms of true positives, true negatives, false positives, and false negatives.
 ```python
smote = SMOTE(random_state=42)
adasyn = ADASYN(random_state=42)

# Define Normalizing Flow
class NormalizingFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])

    def forward(self, x):
        log_det = torch.zeros(x.shape[0])
        for flow in self.flows:
            x = flow(x)
            log_det += torch.slogdet(flow.weight)[1]
        return x, log_det

    def inverse(self, z):
        for flow in reversed(self.flows):
            z = torch.linalg.solve(flow.weight.t(), (z - flow.bias.unsqueeze(0)).t()).t()
        return z

def train_normalizing_flow(X_minority, epochs=100):
    flow = NormalizingFlow(X_minority.shape[1])
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        z, log_det = flow(torch.FloatTensor(X_minority.values))
        prior = MultivariateNormal(torch.zeros_like(z), torch.eye(z.shape[1]))
        loss = -torch.mean(prior.log_prob(z) + log_det)
        loss.backward()
        optimizer.step()
    
    return flow

def oversample_with_normalizing_flow(X_minority, n_samples):
    flow = train_normalizing_flow(X_minority)
    z = torch.randn(n_samples, X_minority.shape[1])
    X_oversampled = flow.inverse(z)
    return X_oversampled.detach().numpy()
```

## Results
The results of each oversampling technique are presented below, including the confusion matrices for a detailed performance breakdown.

```python
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to evaluate model
def evaluate_model(X, y, model, technique_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies, recalls, precisions, f1_scores = [], [], [], []
    confusion_matrices = []

    for train_index, val_index in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Apply oversampling
        if technique_name == 'SMOTE':
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
        elif technique_name == 'ADASYN':
            X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_fold, y_train_fold)
        elif technique_name == 'Normalizing Flows':
            minority_class = X_train_fold[y_train_fold == 1]
            majority_class = X_train_fold[y_train_fold == 0]
            n_minority = len(minority_class)
            n_majority = len(majority_class)
            oversampled_minority = oversample_with_normalizing_flow(minority_class, n_majority - n_minority)
            X_train_resampled = pd.concat([X_train_fold, pd.DataFrame(oversampled_minority, columns=X_train_fold.columns)])
            y_train_resampled = pd.concat([y_train_fold, pd.Series([1] * (n_majority - n_minority))])

        # Train the model and make predictions
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_val_fold)

        # Calculate metrics
        accuracies.append(accuracy_score(y_val_fold, y_pred))
        recalls.append(recall_score(y_val_fold, y_pred))
        precisions.append(precision_score(y_val_fold, y_pred, zero_division=1))
        f1_scores.append(f1_score(y_val_fold, y_pred))

        # Compute and store the confusion matrix
        cm = confusion_matrix(y_val_fold, y_pred)
        confusion_matrices.append(cm)

    # Print overall metrics
    print(f"Results for {technique_name}:")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Mean Recall: {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
    print(f"Mean Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f})")
    print(f"Mean F1-score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

    # Plot confusion matrices
    for i, cm in enumerate(confusion_matrices):
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Fold {i+1} - {technique_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'ConfusionMatrix{i+1}{technique_name}.png')
        plt.show()

    # Final training on the whole set and predictions for the test set
    if technique_name == 'SMOTE':
        X_train_resampled, y_train_resampled = smote.fit_resample(X, y)
    elif technique_name == 'ADASYN':
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X, y)
    elif technique_name == 'Normalizing Flows':
        minority_class = X[y == 1]
        majority_class = X[y == 0]
        n_minority = len(minority_class)
        n_majority = len(majority_class)
        oversampled_minority = oversample_with_normalizing_flow(minority_class, n_majority - n_minority)
        X_train_resampled = pd.concat([X, pd.DataFrame(oversampled_minority, columns=X.columns)])
        y_train_resampled = pd.concat([y, pd.Series([1] * (n_majority - n_minority))])

    model.fit(X_train_resampled, y_train_resampled)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    # Save predictions to CSV
    test_predictions = pd.DataFrame({
        'id': test_data['id'],
        'target': y_pred_test,
        'target_proba': y_pred_proba_test
    })
    test_predictions.to_csv(f'{technique_name}_predictions.csv', index=False)

    print(f"Test predictions saved to {technique_name}_predictions.csv")

# Evaluate models
rf_classifier = RandomForestClassifier(random_state=42)

evaluate_model(X_train, y_train, rf_classifier, 'SMOTE')
evaluate_model(X_train, y_train, rf_classifier, 'ADASYN')
evaluate_model(X_train, y_train, rf_classifier, 'Normalizing Flows')
```

### 1. SMOTE
**Mean Metrics**:
- **Accuracy**: _0.9623_ (± _0.0005_)
- **Recall**: _0.0018_ (± _0.0027_)
- **Precision**: _0.0371_ (± _0.0581_)
- **F1-score**: _0.0035_ (± _0.0051_)

**Confusion Matrices**:
![Confusion Matrix for SMOTE Fold 1](path_to_image_1)
![Confusion Matrix for SMOTE Fold 2](path_to_image_2)
...

### 2. ADASYN
**Mean Metrics**:
- **Accuracy**: _0.9622_ (± _0.0002_)
- **Recall**: _0.0032_ (± _0.0028_)
- **Precision**: _0.0696_ (± _0.0603_)
- **F1-score**: _0.0062_ (± _0.0053_)

**Confusion Matrices**:
![Confusion Matrix for ADASYN Fold 1](path_to_image_3)
![Confusion Matrix for ADASYN Fold 2](path_to_image_4)
...

### 3. Normalizing Flows
**Mean Metrics**:
- **Accuracy**: _0.9636_ (± _0.0000_)
- **Recall**: _0.0000_ (± _0.0000_)
- **Precision**: _1.0000_ (± _0.0000_)
- **F1-score**: _0.0000_ (± _0.0000_)

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
