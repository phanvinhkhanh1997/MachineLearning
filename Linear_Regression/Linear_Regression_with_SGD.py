import numpy as np 
import matplotlib.pyplot as plt 
np.random.seed(1)

data_ = np.genfromtxt('/home/khanh/Desktop/Deep Learning/data.csv',delimiter=",")

data = data_ / 20.0

X_train = data[0:90,0].reshape(90,1)
y_train = data[0:90,1].reshape(90,1)

X_test = data[90:,0]
y_test = data[90:,1]

one = np.ones((X_train.shape[0],1))
X_extend = np.concatenate((one,X_train),axis=1)


def SGD(w_init,learning_rate):
    w = [w_init]
    N = X_extend.shape[0]
    for j in range(100):
        index_i = np.random.permutation(N)
        for i in range(N):
           xi  = X_extend[index_i[i]].reshape(1,2)
           yi  = y_train [index_i[i]].reshape(1,1)  
           new_w = w[-1] - learning_rate * xi.T * (np.dot(xi,w[-1])-yi)
           w.append(new_w)
    return w[-1]
 
w_init = np.array([[0],[1]])
w = SGD(w_init,0.01)

plt.scatter(data[:,0],data[:,1],c='b',s=50)
x = np.linspace(1,5,3)
y = w[0][0] + w[1][0] * x
plt.plot(x,y,c='r',linewidth = 5)
plt.show()