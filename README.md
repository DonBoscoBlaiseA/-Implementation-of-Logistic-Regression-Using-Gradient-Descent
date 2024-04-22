# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries and read the dataset, dropping irrelevant columns and converting categorical variables.
2. Initialize model parameters randomly for further optimization.
3. Define the sigmoid function to compute the sigmoid of given inputs.
4. Define the logistic loss function to measure the model's performance.
5. Implement the gradient descent algorithm to update parameters iteratively and minimize the loss.
6. Train the model using gradient descent, make predictions, and evaluate the model's accuracy.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Don Bosco Blaise A
RegisterNumber: 212221040045
*/

# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# reading the file
dataset = pd.read_csv('Placement_Data.csv')
dataset
```
```
```
```
# dropping the serial no and salary col
dataset = dataset.drop('sl_no',axis = 1)
dataset = dataset.drop('salary',axis = 1)
# categorising col for further labelling
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
# labelling the columns
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
# selecting the features and labels
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
# Initialize the model parameters
theta = np.random.randn(X.shape[1])
y=Y
# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Define the loss function
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
# Define the gradient descent algorithm
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
```
```
        theta -= alpha * gradient
    return theta

# Train the model
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

# Make predictions
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)
# Evaluate the model
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy: ", accuracy)
print(y_pred)
print(Y)
xnew = np.array([0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```
```










```
## Output:
![image](https://github.com/DonBoscoBlaiseA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/140850829/952fa101-48ca-4743-8444-dd011f070626)
![image](https://github.com/DonBoscoBlaiseA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/140850829/f2b35556-6609-4f0d-9860-3032ba449d82)
![image](https://github.com/DonBoscoBlaiseA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/140850829/e43fd578-ca19-4a19-b19e-ce1b9d756632)
![image](https://github.com/DonBoscoBlaiseA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/140850829/4332bb13-a878-49f5-84bd-9ea092ad1c82)
![image](https://github.com/DonBoscoBlaiseA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/140850829/cc00e344-a4fb-4d0c-a66b-2ca9f254afed)
![image](https://github.com/DonBoscoBlaiseA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/140850829/304956b4-9dde-40bc-b151-03120be61617)
![image](https://github.com/DonBoscoBlaiseA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/140850829/fcce257e-f124-489f-bf56-3df2d6f8ebfd)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

