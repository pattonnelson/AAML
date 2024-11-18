# Regression Analysis with SCAD, ElasticNet, and Square Root Lasso Models
## Introduction
This report presents the implementation and comparison of three regularized regression models using PyTorch: the Smoothly Clipped Absolute Deviation (SCAD), ElasticNet, and Square Root Lasso. These models are used to perform regression on datasets with correlated features to evaluate which method best approximates the ideal regression solution.

## Models and Methodology
1. SCAD Regression Model
The SCAD regularization model applies a non-convex penalty that is designed to overcome some limitations of traditional L1 and L2 regularization. It is defined as:
- Penalty Function: The SCAD penalty applies different regularization strengths to coefficients based on their magnitude, aiming to keep large coefficients relatively unshrunken while encouraging sparsity for small coefficients.
2. ElasticNet Model
ElasticNet is a linear regression model that combines both L1 and L2 penalties:

Objective Function:
1
2
MSE
+
𝛼
(
l1_ratio
×
∣
∣
𝑤
∣
∣
1
+
(
1
−
l1_ratio
)
×
1
2
∣
∣
𝑤
∣
∣
2
2
)
2
1
​
 MSE+α(l1_ratio×∣∣w∣∣ 
1
​
 +(1−l1_ratio)× 
2
1
​
 ∣∣w∣∣ 
2
2
​
 )
Use Case: ElasticNet is particularly useful when there are multiple correlated features.
3. Square Root Lasso Model
The Square Root Lasso model is a variant of Lasso regression that minimizes the square root of the mean squared error combined with an L1 penalty. It provides robustness against heteroscedasticity in the data:

Objective Function:
MSE
+
𝛼
∣
∣
𝑤
∣
∣
1
MSE
​
 +α∣∣w∣∣ 
1
​
 
## Data Generation
To test these models, 200 datasets were generated with the following properties:

Number of Samples: 100
Number of Features: 20
Correlation Structure: Features were correlated using a Toeplitz matrix with a correlation coefficient (
𝜌
ρ) of 0.8.
True Coefficients: A predefined coefficient vector, 
𝛽
=
[
−
1
,
2
,
3
,
0
,
0
,
0
,
0
,
2
,
−
1
,
4
]
β=[−1,2,3,0,0,0,0,2,−1,4], was used to generate the true target values.
### Data Generation Function
```python
def make_correlated_features(num_samples, p, rho):
    vcor = [rho**i for i in range(p)]
    r = toeplitz(vcor)
    mu = np.repeat(0, p)
    x = np.random.multivariate_normal(mu, r, size=num_samples)
    return x
```
Training Procedure
For each of the 200 datasets:

Data Conversion: The generated features and targets were converted to PyTorch tensors.
Model Fitting: Each model was trained using the Adam optimizer for a specified number of epochs.
Error Calculation: The Mean Squared Error (MSE) between the predicted and true target values was computed.
Results
The average MSE for each model across the 200 datasets is summarized below:

SCAD Model: 2.1466
ElasticNet Model: 10.2513
Square Root Lasso Model: 10.2938
Conclusion
The SCAD model demonstrated the best performance, with a significantly lower average MSE compared to the ElasticNet and Square Root Lasso models. Thus, the SCAD model appears to approximate the ideal regression solution more closely for this specific setting of correlated features.

Code Implementation
SCAD Regression Model
```python
class SCADRegression(Module):
    ...
```
ElasticNet Model
```python
class ElasticNet(nn.Module):
    ...
```
Square Root Lasso Model
```python
class SqrtLasso(nn.Module):
    ...
```
Data Generation and Model Comparison
```python
# Generate datasets and compare models
...
```
Conclusion
Based on the results, the SCAD model is more effective in this context, likely due to its non-convex penalty, which offers a better balance between sparsity and the preservation of significant coefficients compared to ElasticNet and Square Root Lasso. Future work may involve exploring the models under different data settings, such as varying feature correlations or noise levels.
