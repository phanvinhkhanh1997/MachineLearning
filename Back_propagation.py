import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt
np.random.seed(1)

X,y = datasets.make_moons(400,noise = 0.2)
X_train = X[0:300]
y_train = y[0:300]

X_test = X[300:400]
y_test = y[300:400]

for i in range(len(y_train)):
    if y_train[i]==1:
        plt.scatter(X_train[i,0],X_train[i,1],c='b',s=50)
    elif y_train[i]==0:
        plt.scatter(X_train[i,0],X_train[i,1],c='r',s=50)

plt.show()
W1 = 2 * np.random.randn(3,3) - 1
W2 = 2 * np.random.randn(1,4)  - 1

B1 = np.zeros_like(W1)
B2 = np.zeros_like(W2)
dW1 = 0
dW2 = 0

def g(x):
    return 1/(1+np.exp(-x))

def feed_forward(input,W1,W2):
    one = np.ones((1,1))
    A1  = input
    A1_ = np.concatenate((one[0],A1),axis=0)#add bias +1
    Z2  = np.dot(W1,A1_)
    A2  = g(Z2)
    A2_ = np.concatenate((one[0],A2),axis=0)#add bias +1
    output = g(np.dot(W2,A2_))
    return (A1_,A1,A2_,A2,output)

#print(feed_forward(X_train[0],W1,W2))
def back_ward(input,W1,W2,y_truth):
    (A1_,A1,A2_,A2,output) = feed_forward(input,W1,W2)
    E3 = output - y_truth
    E2 = np.dot(W2.T,E3) * A2_ * (1-A2_)
    E2 = E2[1:4]
    return (E3,E2)

(E3,E2) = back_ward(X_train[0],W1,W2,y_train[0])


def Gradient_Steps(inputs,W1,W2,learning_rate,B1,B2,dW1,dW2):
    for i in range(len(inputs)):
       (A1_,A1,A2_,A2,output) = feed_forward(inputs[i],W1,W2)
       (E3,E2) = back_ward(inputs[i],W1,W2,y_train[i])
       B2 = B2 + np.dot(E3,A2_.reshape(1,4))
       B1 = B1 + np.dot(E2.reshape(3,1),A1_.reshape(1,3))
    #Stochastic_Gradient_Descent
    for i in range(len(inputs)):
       W1 = W1 - learning_rate * B1 / len(inputs)
       W2 = W2 - learning_rate * B2 / len(inputs)
    return (W1,W2)

def Cost_function(inputs,W1,W2):
    s = 0
    for i in range(len(inputs)):
        (A1_,A1,A2_,A2,output) = feed_forward(inputs[i],W1,W2)
        s = s + (y_train[i] * np.log(output) + (1 - y_train[i]) * np.log(1 - output))
    return (-1)*s/len(inputs)     

def Gradient(inputs,W1,W2,learning_rate,n_iterations):
    for i in range(n_iterations):
        (W1,W2) = Gradient_Steps(inputs,W1,W2,learning_rate,B1,B2,dW1,dW2)
        #if i%100 == 0:
         #   print(Cost_function(inputs,W1,W2))
    return (W1,W2)

def predict(input,W1,W2):
    one = np.ones((1,1))
    A1  = input
    A1_ = np.concatenate((one[0],A1),axis=0)#add bias +1
    Z2  = np.dot(W1,A1_)
    A2  = g(Z2)
    A2_ = np.concatenate((one[0],A2),axis=0)#add bias +1
    output = g(np.dot(W2,A2_))
    return output
(W1,W2) = Gradient(X_train,W1,W2,0.001,100000)

for i in range(100):
    print("predict : %.2f <===> target : %.2f\n" % (predict(X_test[i],W1,W2),y_test[i]))
