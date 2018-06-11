import numpy as np 
import matplotlib.pyplot as plt 
np.random.seed(1)

data = np.genfromtxt('/home/khanh/Desktop/Deep Learning/data.csv',delimiter = ",")

m_initial = np.random.randn()
b_initial = np.random.randn()
 

def Gradient_Steps(data,m_current,b_current,learning_rate):
    N = len(data)
    dm = 0
    db = 0
    for i in range(N):
        x = data[i,0]
        y = data[i,1]
        dm += -(1/N) * x * (y-(m_current*x+b_current))
        db += -(1/N) * (y-(m_current*x+b_current))

    for i in range(N):    
       m = m_current -  learning_rate * dm 
       b = b_current -  learning_rate * db 
    return [m,b]

def Gradient(data,m,b,learning_rate,n_iterations):
    for i in range(n_iterations):
       m,b = Gradient_Steps(data,m,b,learning_rate)
    return [m,b]

[m,b] = Gradient(data,m_initial,b_initial,0.0001,1000)

plt.scatter(data[:,0],data[:,1],c='b',s=50)
X0 = np.linspace(0,100,2)
Y0 = m*X0 + b
plt.plot(X0,Y0,c='r',linewidth=2)
plt.show()