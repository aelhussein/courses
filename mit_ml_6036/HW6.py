# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:30:17 2021

@author: ahmed
"""
import math
import numpy as np
# =============================================================================
# HW5
# =============================================================================

# Hinge loss gradient
def hinge_loss_grad(x, y, a):
  return np.where(y*a>1,0,-y*x)

# Softmax function
def SM(z):
    return (np.exp(z)/np.sum(np.exp(z))).T

# Softmax cacluation
w = np.array([[1, -1, -2], [-1, 2, 1]])
x = np.array([[1], [1]])
y = np.array([[0, 1, 0]]).T

#Calculate a
a = np.dot(w.T,x)
#Get Softmax probs
z_a = SM(a)
#Get diff w.r.t w
d_w = x*(z_a-y).T
#Update w with gd
w2 = w - 0.5*d_w
a2 = np.dot(w2.T,x)
z_a = SM(a2)

#Problem 3
#Neural network

# layer 1 weights
w_1 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
w_1_bias = np.array([[-1, -1, -1, -1]]).T
# layer 2 weights
w_2 = np.array([[1, -1], [1, -1], [1, -1], [1, -1]])
w_2_bias = np.array([[0, 2]]).T

x = np.array([[3],[14]])
z_1 = np.dot(w_1.T, x) + w_1_bias
a_1 = np.where(z_1>0,z_1,0)

z_2 = np.dot(w_2.T, a_1) + w_2_bias
a_2 = SM(z_2)

SM(np.array([[3],[-1]]))
