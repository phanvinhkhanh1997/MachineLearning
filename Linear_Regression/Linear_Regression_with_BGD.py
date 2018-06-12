import numpy as np 
import matplotlib.pyplot as plt 
np.random.seed(1)
#Prepare datasets
data = np.genfromtxt('/home/khanh/Desktop/Deep Learning/data.csv',delimiter=",")

#Splitting data
X_train = data[0:90,0]
y_train = data[0:90,1]

X_test = data[90:,0]
y_test = data[90:,1]

#Reshape Training set
X_train = X_train.reshape((90,1))
y_train = y_train.reshape((90,1))

#X_extend
one = np.ones((X_train.shape[0],1))
X_extend = np.concatenate((one,X_train),axis=1)

def loss_function(A):
    N = X_extend.shape[0]
    return (0.5/N) * np.linalg.norm(np.dot(X_extend,A) - y_train)**2

def derivative(A):  
    N = X_extend.shape[0]
    return (1/N) * np.dot(X_extend.T,np.dot(X_extend,A) - y_train)

def numeric_derivative(A):
    eps = 1e-3
    g = np.zeros_like(A)
    for i in range(len(A)):
        A_1 = A.copy()
        A_2 = A.copy()
        A_1[i] += eps
        A_2[i] -= eps
        g[i] = (loss_function(A_1) - loss_function(A_2)) / (2 * eps)
    return g    

def checking_derivative(A):
    value1 = derivative(A)
    value2 = numeric_derivative(A)
    if np.linalg.norm(value1 - value2) < 1e-5:
        return True
    else :
        return False

print("Checking Derivative ===>",checking_derivative(np.random.rand(2,1).reshape(2,1)))           

def Gradient_Descent(A0,learning_rate):
    A = [A0]
    for i in range(100):
           new_A = A[-1] - learning_rate * derivative(A[-1])
           if np.linalg.norm(derivative(new_A)) / len(new_A) < 1e-3:
              break
           A.append(new_A)
    return A[-1]

A_init = np.array([[5],[1]])
A = Gradient_Descent(A_init,0.01)

W = np.dot(np.linalg.pinv(np.dot(X_extend.T,X_extend)),np.dot(X_extend.T,y_train))
W_0 = W[0][0]
W_1 = W[1][0]
plt.scatter(data[:,0],data[:,1],c='b',s=50)
x = np.linspace(0,100,3,endpoint = True)
y = W_0 + W_1 * x 
plt.plot(x,y,c='y',linewidth = 5)
plt.show()
#print(A[0][0])

#print(W)
