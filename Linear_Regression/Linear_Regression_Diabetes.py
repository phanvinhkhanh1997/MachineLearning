import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets 

#Prepare datasets
data = datasets.load_diabetes()

#Splitting datasets
X_train = data.data[0:400].reshape((400,10))
y_train = data.target[0:400].reshape((400,1))

X_test = data.data[400:].reshape((42,10))
y_test = data.target[400:].reshape((42,1))

#X_train_extend
one = np.ones((400,1))
X_train_extend = np.concatenate((one,X_train),axis=1)

#Optimize A
A = np.dot(np.linalg.pinv(np.dot(X_train_extend.T,X_train_extend)),np.dot(X_train_extend.T,y_train))

#X_test_extend
one2 = np.ones((X_test.shape[0],1))
X_test_extend = np.concatenate((one2,X_test),axis=1)

#Test with Testing datasets
Result = np.dot(X_test_extend,A)
for i in range(42):
   print("Predict : %.2f  <===> Target : %.2f" % (Result[i],y_test[i]))