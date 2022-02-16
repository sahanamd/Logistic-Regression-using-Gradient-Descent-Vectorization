# Logistic-Regression-using-Gradient-Descent-Vectorization
Logistic Regression in Python by using Vectorized and Gradient Descent
# Gradient Descent including threshold on error

import time
X = Vector_X.values
Y = df['Pass (yi)'].values
Beta = [0, 0]

count = len(X)
cost_val = []
beta_val = []
TSH = np.power(10.,-10)
# l = 0.3
l = 0.4
Cost_val = []
beta_val = []
Z = Beta @ np.transpose(X)
Sigmoid = 1/ (1 + np.exp(-Z))
first_der = X.T @ (Sigmoid - Y)
i = 0

for j in range(-1,-11,-1):
    TSH = np.power(10.,j)
    Cost_val = []
    beta_val = []
    Beta = [0, 0]
    Z = Beta @ np.transpose(X)
    Sigmoid = 1/ (1 + np.exp(-Z))
    first_der = X.T @ (Sigmoid - Y)
    i = 0
    start_time = time.time()
#     print(j)
    while np.linalg.norm(first_der) > TSH:
        i += 1
        Z = Beta @ np.transpose(X)
        Sigmoid = 1/ (1 + np.exp(-Z))
        Cost = -1/count * (np.transpose(Y) @ np.log10(Sigmoid) + np.transpose(1 - Y) @ np.log10(1 - Sigmoid))
        first_der = X.T @ (Sigmoid - Y)
        Beta = Beta - (l/count * X.T @ (Sigmoid - Y))
        Cost_val.append(Cost)
        beta_val.append(Beta)
    
    end_time = time.time()
    Total_Time = end_time - start_time
    print("The loss is ", np.linalg.norm(first_der))
    print("Number of iterations : ", i)
    print("Time taken to execute the SGD (in sec):", Total_Time)
    print("Co-efficients are : ", Beta)
    plt.plot(range(i),Cost_val)
    plt.show()
    time.sleep(1)



# Gradient Descent by Newton's method

import matplotlib.pyplot as plt 
import time
import pandas as pd
import numpy as np

def f(beta):
  return np.ravel(np.ones(len(Y))*(np.log(1+np.exp(X*beta)))-Y.T*X*beta)[0]
def nabla_f(beta):
  return X.T*(1/(1+1/np.exp(X*beta))-Y)
def nabla2_f(beta):
  return X.T*(np.diag(np.ravel(np.exp(X*beta)/np.power(1+np.exp(X*beta),2)))*X)


df = pd.read_excel(r'C:\Logistic_Regression\data\Logistic_Regression_Data.xlsx')
Vector_X = df.drop(columns=['Pass (yi)'])
Vector_X.insert(0, 'X0', 1)
Vector_X.head()
X = np.matrix(Vector_X.values)
Y = df['Pass (yi)'].values
Y = np.matrix(Y.reshape(-1,1))

beta = np.matrix(np.zeros(X.shape[1])).T
TOL = np.power(10.,-10)
counter = 0
beta_0 = beta[0]
beta_1 = beta[1]

Hessian = X.T * np.diag(np.ravel(np.exp(X*beta)/np.power(1+np.exp(X*beta),2))) * X
first_der = X.T*(1/(1+1/np.exp(X*beta))-Y)
cost_val = [np.linalg.norm(first_der)]

start_time = time.time()
while np.linalg.norm(first_der) > TOL:

    counter += 1
    Hessian = X.T * np.diag(np.ravel(np.exp(X*beta)/np.power(1+np.exp(X*beta),2))) * X
    first_der = X.T*(1/(1+1/np.exp(X*beta))-Y)
    beta = beta - np.linalg.inv(Hessian)*(first_der)
    beta_0 = np.concatenate((beta_0, beta[0]))
    beta_1 = np.concatenate((beta_1, beta[1]))
    cost = np.linalg.norm(first_der)
    cost_val.append(cost)
    

end_time = time.time()
Total_Time = end_time - start_time  

print("The loss is ", np.linalg.norm(first_der))
print("Number of iterations : ", counter)
print("Time taken to execute the SGD (in sec):", Total_Time)
print("Co-efficients are : ", Beta)
plt.plot(range(counter+1), cost_val)

