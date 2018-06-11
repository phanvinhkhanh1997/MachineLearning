import numpy as np 

def f(x):
    return x**2 + 5 * np.sin(x)

def f_prime(x):
    return 2 * x + 5 * np.cos(x)

def Gradient_Descent(x0,learning_rate):
    x = [x0]
    for i in range(1000):
        new_x = x[-1] - learning_rate * f_prime(x[-1])
        if abs(f_prime(new_x))<1e-5:
            break
        x.append(new_x) 
    return x[-1] 

x0 = Gradient_Descent(-1,0.01)    
x1 = Gradient_Descent(3 ,0.01)

print(f(x0))
print(f(x1))