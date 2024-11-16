import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

# Load pre-split data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Subset 10% of observations from each class in the training data
train_data_subset = train_data.groupby('target').apply(lambda x: x.sample(frac=0.1, random_state=42)).reset_index(drop=True)

# Separate features and target for the training data
X_train = train_data_subset.drop(['id', 'target'], axis=1)
y_train = train_data_subset['target']

# Prepare test data (without target column)
X_test = test_data.drop('id', axis=1)

# Define oversampling techniques
smote = SMOTE(random_state=42)
adasyn = ADASYN(random_state=42)

# Normalizing Flows implementation

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

# Function to evaluate model
def evaluate_model(X, y, model, technique_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    recalls = []
    precisions = []
    f1_scores = []
    
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
        
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_val_fold)
        
        accuracies.append(accuracy_score(y_val_fold, y_pred))
        recalls.append(recall_score(y_val_fold, y_pred))
        precisions.append(precision_score(y_val_fold, y_pred))
        f1_scores.append(f1_score(y_val_fold, y_pred))
    
    print(f"Results for {technique_name}:")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Mean Recall: {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
    print(f"Mean Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f})")
    print(f"Mean F1-score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    
    # Train on full training set and generate predictions for test set
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