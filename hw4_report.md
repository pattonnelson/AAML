# Comparison of Oversampling Techniques for Imbalanced Classification
## Introduction
This report outlines the implementation and comparison of three oversampling techniques used to address class imbalance in the Porto Seguro Safe Driver Prediction dataset. The techniques compared are:
1. Synthetic Minority Oversampling Technique (SMOTE)
2. Adaptive Synthetic (ADASYN) Sampling
3. Normalizing Flows
The goal is to evaluate these techniques' effectiveness in improving classification performance on imbalanced data.
## Dataset
The Porto Seguro Safe Driver Prediction dataset is used for this analysis. Due to its large size, we work with a subset containing 10% of observations from each class. The data is pre-split into training and test sets, with the training set containing the target variable and the test set excluding it.
```python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Subset 10% of observations from each class in the training data
train_data_subset = train_data.groupby('target').apply(lambda x: x.sample(frac=0.1, random_state=42)).reset_index(drop=True)

X_train = train_data_subset.drop(['id', 'target'], axis=1)
y_train = train_data_subset['target']

X_test = test_data.drop('id', axis=1)
```
## Oversampling Techniques
1. SMOTE (Synthetic Minority Oversampling Technique)
SMOTE works by creating synthetic examples in the feature space. For each minority class sample, it finds its k-nearest neighbors and creates new samples by interpolating between the sample and its neighbors.
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
```
2. ADASYN (Adaptive Synthetic Sampling)
ADASYN is similar to SMOTE but focuses on generating samples in the areas where the minority class samples are harder to learn. It does this by generating more synthetic data for minority class samples that are harder to learn, based on their proximity to the majority class.
```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42)
```
3. Normalizing Flows
Normalizing Flows is a more advanced technique that learns a complex, invertible transformation between a simple base distribution and the data distribution. This allows for generating new samples by sampling from the base distribution and applying the inverse transformation.
```python
class NormalizingFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(3)  # Simple example with linear flows
        ])

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
    optimizer = torch.optim.Adam(flow.parameters())
    
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
## Evaluation Procedure
The evaluation procedure for each oversampling technique follows these steps:<br>
1. Perform stratified 5-fold cross-validation on the training data:
- For each fold:
  - Apply the oversampling technique to the training fold
  - Train a Random Forest classifier on the oversampled data
  - Evaluate the model on the validation fold
- Calculate mean and standard deviation of performance metrics (accuracy, recall, precision, F1-score)
2. Train a final model on the entire oversampled training set
3. Generate predictions for the test set
4. Save the test predictions to a CSV file
```python
def evaluate_model(X, y, model, technique_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies, recalls, precisions, f1_scores = [], [], [], []
    
    for train_index, val_index in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        # oversampling
        if technique_name == 'SMOTE':
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
        elif technique_name == 'ADASYN':
            X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_fold, y_train_fold)
        elif technique_name == 'Normalizing Flows':
            minority_class = X_train_fold[y_train_fold == 1]
            majority_class = X_train_fold[y_train_fold == 0]
            n_minority, n_majority = len(minority_class), len(majority_class)
            oversampled_minority = oversample_with_normalizing_flow(minority_class, n_majority - n_minority)
            X_train_resampled = pd.concat([X_train_fold, pd.DataFrame(oversampled_minority, columns=X_train_fold.columns)])
            y_train_resampled = pd.concat([y_train_fold, pd.Series([1] * (n_majority - n_minority))])
        
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_val_fold)
        
        accuracies.append(accuracy_score(y_val_fold, y_pred))
        recalls.append(recall_score(y_val_fold, y_pred))
        precisions.append(precision_score(y_val_fold, y_pred))
        f1_scores.append(f1_score(y_val_fold, y_pred))
    
    # cross-validation results
    print(f"Results for {technique_name}:")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Mean Recall: {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
    print(f"Mean Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f})")
    print(f"Mean F1-score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    
    # generate predictions for test set
    X_train_resampled, y_train_resampled = apply_oversampling(X, y, technique_name)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    
    test_predictions = pd.DataFrame({
        'id': test_data['id'],
        'target': y_pred_test,
        'target_proba': y_pred_proba_test
    })
    test_predictions.to_csv(f'{technique_name}_predictions.csv', index=False)
    
    print(f"Test predictions saved to {technique_name}_predictions.csv")

# Evaluate
rf_classifier = RandomForestClassifier(random_state=42)

evaluate_model(X_train, y_train, rf_classifier, 'SMOTE')
evaluate_model(X_train, y_train, rf_classifier, 'ADASYN')
evaluate_model(X_train, y_train, rf_classifier, 'Normalizing Flows')
```
## Comparison of Techniques
### SMOTE
SMOTE is a well-established technique that creates synthetic samples by interpolating between existing minority class samples. It's relatively simple to implement and understand, and often provides good results.
### ADASYN
ADASYN is an extension of SMOTE that focuses on generating more synthetic samples for minority class instances that are harder to learn. This can be beneficial when the minority class has complex decision boundaries.
### Normalizing Flows
Normalizing Flows is a more advanced technique that learns a complex transformation of the data distribution. It has the potential to capture more nuanced patterns in the data, but it's also more computationally intensive and may require more tuning to achieve optimal results.
## Conclusion
By comparing these three oversampling techniques, we can gain insights into their relative performance on the Porto Seguro Safe Driver Prediction dataset. The cross-validation results provide a robust estimate of each technique's effectiveness, while the test set predictions allow for submission to the Kaggle competition for external validation.
When interpreting the results, consider not only the performance metrics but also the computational complexity and ease of implementation of each technique. The choice of the best oversampling method may depend on the specific characteristics of the dataset and the requirements of the project.
