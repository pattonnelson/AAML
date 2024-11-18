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
/
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
/
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

- Number of Samples: 100
- Number of Features: 20
- Correlation Structure: Features were correlated using a Toeplitz matrix with a correlation coefficient (𝜌) of 0.8.
- True Coefficients: A predefined coefficient vector, 
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
] was used to generate the true target values.
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
    def __init__(self, input_size, lambda_=1.0, a=3.7, fit_intercept=True):
        """
        SCAD regularized regression.
        
        Parameters:
            input_size (int): Number of input features.
            lambda_ (float): SCAD regularization strength.
            a (float): SCAD tuning parameter (default=3.7).
            fit_intercept (bool): Whether to fit an intercept term.
        """
        super(SCADRegression, self).__init__()
        self.input_size = input_size
        self.lambda_ = lambda_
        self.a = a
        
        # Define the linear regression layer
        self.linear = Linear(input_size, 1, bias=fit_intercept, device='cpu', dtype=torch.float64)

    def forward(self, x):
        return self.linear(x)

    def scad_penalty(self, weights):
        """
        Computes the SCAD penalty for given weights.

        Parameters:
            weights (Tensor): Tensor of weights to apply regularization to.

        Returns:
            Tensor: SCAD penalty.
        """
        abs_weights = weights.abs()
        scad_penalty = torch.zeros_like(abs_weights)

        # Define regions for SCAD penalty
        mask_1 = (abs_weights <= self.lambda_)
        mask_2 = (abs_weights > self.lambda_) & (abs_weights <= self.a * self.lambda_)
        mask_3 = (abs_weights > self.a * self.lambda_)

        # Apply SCAD formula per region
        scad_penalty[mask_1] = self.lambda_ * abs_weights[mask_1]
        scad_penalty[mask_2] = (-abs_weights[mask_2]**2 + 2 * self.a * self.lambda_ * abs_weights[mask_2] - self.lambda_**2) / (2 * (self.a - 1))
        scad_penalty[mask_3] = (self.lambda_**2 * (self.a + 1)) / 2

        return scad_penalty.sum()

    def objfunc(self, y_pred, y_true):
        """
        Computes the objective function including the SCAD penalty.

        Parameters:
            y_pred (Tensor): Model predictions.
            y_true (Tensor): True labels.

        Returns:
            Tensor: Objective function value.
        """
        mse_loss = MSELoss()(y_pred, y_true)
        scad_reg = self.scad_penalty(self.linear.weight)

        objective = (1 / 2) * mse_loss + self.lambda_ * scad_reg
        return objective

    def fit(self, X, y, num_epochs=1000, learning_rate=0.01):
        """
        Fits the model to the data using gradient descent.

        Parameters:
            X (Tensor): Input features.
            y (Tensor): Target values.
            num_epochs (int): Number of epochs.
            learning_rate (float): Learning rate.
        """
        optimizer = Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train() 
            optimizer.zero_grad()
            y_pred = self(X)
            obj_val = self.objfunc(y_pred, y)
            loss = MSELoss()(y_pred, y)  # Regular MSE for reporting
            obj_val.backward()
            optimizer.step()

            #if (epoch + 1) % 100 == 0:
                #print(f"Epoch [{epoch + 1}/{num_epochs}], MSE: {loss.item()}")

    def predict(self, X):
        """
        Predicts target values using the trained model.

        Parameters:
            X (Tensor): Input features.

        Returns:
            Tensor: Predicted values.
        """
        self.eval()
        with torch.no_grad():
            y_pred = self(X)
        return y_pred

    def get_coefficients(self):
        """
        Returns the learned model coefficients.

        Returns:
            Tensor: Model coefficients.
        """
        return self.linear.weight    ...
```
ElasticNet Model
```python
class ElasticNet(nn.Module):
    def __init__(self, input_size, alpha=1.0, l1_ratio=0.5):
        """
        Initialize the ElasticNet regression model.

        Args:
            input_size (int): Number of input features.
            alpha (float): Regularization strength. Higher values of alpha
                emphasize L1 regularization, while lower values emphasize L2 regularization.
            l1_ratio (float): The ratio of L1 regularization to the total
                regularization (L1 + L2). It should be between 0 and 1.

        """
        super(ElasticNet, self).__init__()
        self.input_size = input_size
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        # Define the linear regression layer
        self.linear = nn.Linear(input_size, 1,bias=False,device=device,dtype=dtype)

    def forward(self, x):
        """
        Forward pass of the ElasticNet model.

        Args:
            x (Tensor): Input data with shape (batch_size, input_size).

        Returns:
            Tensor: Predicted values with shape (batch_size, 1).

        """
        return self.linear(x)

    def loss(self, y_pred, y_true):
        """
        Compute the ElasticNet loss function.

        Args:
            y_pred (Tensor): Predicted values with shape (batch_size, 1).
            y_true (Tensor): True target values with shape (batch_size, 1).

        Returns:
            Tensor: The ElasticNet loss.

        """
        mse_loss = nn.MSELoss()(y_pred, y_true)
        l1_reg = torch.norm(self.linear.weight, p=1)
        l2_reg = torch.norm(self.linear.weight, p=2)

        objective = (1/2) * mse_loss + self.alpha * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * (1/2)*l2_reg**2)

        return objective

    def fit(self, X, y, num_epochs=100, learning_rate=0.01):
        """
        Fit the ElasticNet model to the training data.

        Args:
            X (Tensor): Input data with shape (num_samples, input_size).
            y (Tensor): Target values with shape (num_samples, 1).
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimization.

        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X)
            loss = self.loss(y_pred, y)
            loss.backward()
            optimizer.step()

            #if (epoch + 1) % 100 == 0:
               # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    def predict(self, X):
        """
        Predict target values for input data.

        Args:
            X (Tensor): Input data with shape (num_samples, input_size).

        Returns:
            Tensor: Predicted values with shape (num_samples, 1).

        """
        self.eval()
        with torch.no_grad():
            y_pred = self(X)
        return y_pred
    def get_coefficients(self):
        """
        Get the coefficients (weights) of the linear regression layer.

        Returns:
            Tensor: Coefficients with shape (output_size, input_size).

        """
        return self.linear.weight    ...
```
Square Root Lasso Model
```python
class SqrtLasso(nn.Module):
    def __init__(self, input_size, alpha=0.1):
        """
        Initialize the  regression model.


        """
        super(SqrtLasso, self).__init__()
        self.input_size = input_size
        self.alpha = alpha


        # Define the linear regression layer
        self.linear = nn.Linear(input_size, 1,bias=False,device=device,dtype=dtype)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input data with shape (batch_size, input_size).

        Returns:
            Tensor: Predicted values with shape (batch_size, 1).

        """
        return self.linear(x)

    def loss(self, y_pred, y_true):
        """
        Compute the loss function.

        Args:
            y_pred (Tensor): Predicted values with shape (batch_size, 1).
            y_true (Tensor): True target values with shape (batch_size, 1).

        Returns:
            Tensor: The loss.

        """
        mse_loss = nn.MSELoss(reduction='mean')(y_pred, y_true)  # Use 'mean' instead of 'mse'
        l1_reg = torch.norm(self.linear.weight, p=1, dtype=torch.float64)

        # Square Root Lasso loss
        loss = torch.sqrt(mse_loss) + self.alpha * l1_reg
        return loss

    def fit(self, X, y, num_epochs=200, learning_rate=0.01):
        """
        Fit the model to the training data.

        Args:
            X (Tensor): Input data with shape (num_samples, input_size).
            y (Tensor): Target values with shape (num_samples, 1).
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimization.

        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X)
            loss = self.loss(y_pred, y)
            loss.backward()
            optimizer.step()

            #if (epoch + 1) % 100 == 0:
                #print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    def predict(self, X):
        """
        Predict target values for input data.

        Args:
            X (Tensor): Input data with shape (num_samples, input_size).

        Returns:
            Tensor: Predicted values with shape (num_samples, 1).

        """
        self.eval()
        with torch.no_grad():
            y_pred = self(X)
        return y_pred
    def get_coefficients(self):
        """
        Get the coefficients (weights) of the linear regression layer.

        Returns:
            Tensor: Coefficients with shape (output_size, input_size).

        """
        return self.linear.weight  
```
Data Generation and Model Comparison
```python
# Parameters
num_datasets = 200      # Number of datasets
num_samples = 100       # Number of samples per dataset
num_features = 20       # Number of features per dataset
rho = 0.8               # Correlation coefficient for features
beta = np.array([-1, 2, 3, 0, 0, 0, 0, 2, -1, 4]).reshape(-1, 1)
betastar = np.concatenate([beta, np.repeat(0, num_features - len(beta)).reshape(-1, 1)], axis=0)


# Initialize lists to store errors for each model
scad_errors = []
elasticnet_errors = []
sqrt_lasso_errors = []

for i in range(num_datasets):
    # Generate features and target for the dataset
    x = make_correlated_features(num_samples, num_features, rho)
    y = x @ betastar + 1.5 * np.random.normal(size=(num_samples, 1))

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(x, dtype=dtype, device=device)
    y_tensor = torch.tensor(y, dtype=dtype, device=device)...
```
## Conclusion
Average MSE for SCAD model: 2.1466109223818406<br><br>
Average MSE for ElasticNet model: 10.251284280684455<br><br>
Average MSE for Square Root Lasso model: 10.29377798661154<br><br>
SCAD model approximates the ideal solution more closely.<br><br>
Based on the results, the SCAD model is more effective in this context, likely due to its non-convex penalty, which offers a better balance between sparsity and the preservation of significant coefficients compared to ElasticNet and Square Root Lasso. Future work may involve exploring the models under different data settings, such as varying feature correlations or noise levels.
