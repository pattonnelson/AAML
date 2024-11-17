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
1.Perform stratified 5-fold cross-validation on the training data:
- For each fold:
  - Apply the oversampling technique to the training fold
  - Train a Random Forest classifier on the oversampled data
  - Evaluate the model on the validation fold
- Calculate mean and standard deviation of performance metrics (accuracy, recall, precision, F1-score)
2. Train a final model on the entire oversampled training set
3. Generate predictions for the test set
4. Save the test predictions to a CSV file
