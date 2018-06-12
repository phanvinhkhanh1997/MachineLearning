import numpy as np 
import matplotlib.pyplot as plt 

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

#Optimize A
A = np.dot(np.linalg.pinv(np.dot(X_extend.T,X_extend)),np.dot(X_extend.T,y_train))

#Resutl Optimize A
A_0 = A[0][0]
A_1 = A[1][0]

#Visualize data
plt.scatter(data[:,0],data[:,1],c='b',s=50)
X = np.linspace(2,100,2)
Y = A_0 + A_1  * X 
plt.plot(X,Y,c='r',linewidth = 2)
plt.show()

#Test with Training set
for i in range(10):
	print("Predict : %.2f <====> Target : %.2f\n" % ((A_0 + A_1*X_test[i]),y_test[i]))
