# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:42:59 2021

@author: ahmed
"""
import numpy as np


#Calculate the Margin
def margin(theta, theta0, x, y):
    return y*(theta.T.dot(x)+theta0)/np.linalg.norm(theta)
    

#Get the margins
data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
blue_margin = margin(blue_th, blue_th0, data, labels)
blue_margins = [np.sum(blue_margin), np.min(blue_margin), np.max(blue_margin)]



red_th = np.array([[1, 0]]).T
red_th0 = -2.5
red_margin = margin(red_th, red_th0, data, labels)
red_margins = [np.sum(red_margin), np.min(red_margin), np.max(red_margin)]

#Calculate y/yref (needed for hinge loss)
data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4
margins = margin(th, th0, data, labels)
yref = (2**0.5)/2
1 - margins/yref


# =============================================================================
# Gradient Descent
# =============================================================================
#Helper functions
def rv(value_list):
    return np.array([value_list])

def cv(value_list):
    return np.transpose(rv(value_list))

def f1(x):
    return float((2 * x + 3)**2)

def df1(x):
    return 2 * 2 * (2 * x + 3)

def f2(v):
    x = float(v[0]); y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def df2(v):
    x = float(v[0]); y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])

def step_size_fn(i):
    return 0.01

x0 = cv([1,2,3,4])


def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]

#Functions

def gd(f, df, x0, step_size_fn, max_iter):
    fs =[f(x0)]
    xs = [x0]
    for i in range(max_iter):
        x0 = x0 - step_size_fn(i)*(df(x0))
        fs.append(f(x0))
        xs.append(x0)
    return (x0, fs, xs)

#Numerical gradient        
def num_grad(f,delta = 0.001):
    def grad(x):
        gradient = np.zeros(len(x))
        for i in range(len(x)):
            x_up, x_down = x.copy(), x.copy()
            x_up[i] += delta
            x_down[i] -= delta
            gradient[i] = (f(x_up)- f(x_down))/(2*delta)
        gradient = gradient.reshape(-1,1)
        return gradient
    return grad

# Using numerical gradient in gradient descent
def minimize(f, x0, step_size_fn, max_iter):
    fs =[f(x0)]
    xs = [x0]
    for i in range(max_iter):
        x0 = x0 - step_size_fn(i)*(num_grad(f)(x0))
        fs.append(f(x0))
        xs.append(x0)
    return (x0, fs, xs)
        
minimize(f1, x0, step_size_fn , 100)        
 x0 = cv([0.,0.])       
max_iter = 100  
minimize(f2, x0, step_size_fn , 100) 


# SVM implementation
# Helper function
def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

#Calculating losses
def hinge(v):
    return np.where(v<1,1-v,0)

def hinge_loss(x, y, th, th0):
    loss = y*(np.dot(th.T, x) + th0)
    return hinge(loss)    

def svm_obj(x, y, th, th0, lam):
    loss = hinge_loss(x,y,th, th0)
    return np.mean(loss) + lam*(np.linalg.norm(th)**2)

#Calculating gradients
def d_hinge(v):
    return np.where(v<1, -1, 0)

def d_hinge_loss_th(x, y, th, th0):
    d_loss = d_hinge(y*(np.dot(th.T, x) + th0))*y*x
    return d_loss

def d_hinge_loss_th0(x, y, th, th0):
    return d_hinge(y*(np.dot(th.T,x)+th0))*y

def d_svm_obj_th(x, y, th, th0, lam):
    d_loss_th = d_hinge_loss_th(x, y, th, th0)
    return np.mean(d_loss_th) + 2*lam*th

def d_svm_obj_th0(x, y, th, th0, lam):
    d_loss_th0 = d_hinge_loss_th0(x, y, th, th0)
    return np.array(np.mean(d_loss_th0)).reshape(1,1)

def svm_obj_grad(X, y, th, th0, lam):
    d_loss_th = d_svm_obj_th(x, y, th, th0, lam)
    d_loss_th0 = d_svm_obj_th0(x, y, th, th0, lam)
    return np.vstack((d_loss_th, d_loss_th0))

#Test cases
X1 = np.array([[1, 2, 3, 9, 10]])
y1 = np.array([[1, 1, 1, -1, -1]])
th1, th10 = np.array([[-0.31202807]]), np.array([[1.834     ]])
X2 = np.array([[2, 3, 9, 12],
               [5, 2, 6, 5]])
y2 = np.array([[1, -1, 1, -1]])
th2, th20=np.array([[ -3.,  15.]]).T, np.array([[ 2.]])

d_hinge(np.array([[ 71.]])).tolist()
d_hinge(np.array([[ -23.]])).tolist()
d_hinge(np.array([[ 71, -23.]])).tolist()

d_hinge_loss_th(X2[:,0:1], y2[:,0:1], th2, th20).tolist()
d_hinge_loss_th(X2, y2, th2, th20).tolist()
d_hinge_loss_th0(X2[:,0:1], y2[:,0:1], th2, th20).tolist()
d_hinge_loss_th0(X2, y2, th2, th20).tolist()

d_svm_obj_th(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()
d_svm_obj_th(X2, y2, th2, th20, 0.01).tolist()
d_svm_obj_th0(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()
d_svm_obj_th0(X2, y2, th2, th20, 0.01).tolist()

svm_obj_grad(X2, y2, th2, th20, 0.01).tolist()
svm_obj_grad(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()



# Batch SVM GD
def batch_svm_min(data, labels, lam):
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    init = np.zeros((data.shape[0] + 1, 1))

    def f(th):
      return svm_obj(data, labels, th[:-1, :], th[-1:,:], lam)

    def df(th):
      return svm_obj_grad(data, labels, th[:-1, :], th[-1:,:], lam)

    x, fs, xs = gd(f, df, init, svm_min_step_size_fn, 10)
    return x, fs, xs

#Test
def separable_medium():
    X = np.array([[2, -1, 1, 1],
                  [-2, 2, 2, -1]])
    y = np.array([[1, -1, 1, -1]])
    return X, y
sep_m_separator = np.array([[ 2.69231855], [ 0.67624906]]), np.array([[-3.02402521]])

x_1, y_1 = super_simple_separable()
ans = package_ans(batch_svm_min(x_1, y_1, 0.0001))

x_1, y_1 = separable_medium()
ans = package_ans(batch_svm_min(x_1, y_1, 0.0001))
x_1.shape
np.zeros((2,1))











