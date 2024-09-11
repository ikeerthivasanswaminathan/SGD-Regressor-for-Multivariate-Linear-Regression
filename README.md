# SGD Regressor for Multivariate Linear Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Data Preparation
3. Hypothesis DefinitionCost Function
4. Parameter Update Rule
5. Iterative Training
6. Model Evaluation
7. End 

## Program:

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.

Developed by : KEERTHIVASAN S

RegisterNumber : 212223220046

```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()

X = data.data[:, :3]

Y=np.column_stack((data.target,data.data[:, 6]))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()

scaler_X = StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

sgd=SGDRegressor(max_iter=1000, tol=1e-3)

multi_output_sgd=MultiOutputRegressor(sgd)

multi_output_sgd.fit(X_train,Y_train)

Y_pred=multi_output_sgd.predict(X_test)

Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)

print("\nPredictions:\n",Y_pred[:5])
```

## Output:

![exp4op1](https://github.com/user-attachments/assets/e132564b-83cb-4c78-adcc-47742d21cf04)

![exp4op2](https://github.com/user-attachments/assets/dbb5e5be-a170-4d02-9531-782b56745799)

![exp4op3](https://github.com/user-attachments/assets/180d033c-2f45-473b-a020-62f3b5d7b4b2)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
