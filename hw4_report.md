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
