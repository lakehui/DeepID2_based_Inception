#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:44:05 2017

@author: hh
"""

from keras import backend as K
import numpy as np

#def Solve():  
    
K.set_epsilon(1e-11)
x = K.placeholder(shape=(None, 1))
y = K.placeholder(shape=(None, 1))
label = K.placeholder(shape=(None, 1))

#x= K.variable(value = X_train)
#y = K.variable(value = Y_train)
#label = K.variable(value = L_train)

W = K.variable(value = 0.5)#np.random.rand(1))
b = K.variable(value = 1.0)#np.random.rand(1))
U = K.square(W*x + b)

lamda = K.variable(value = 1.0)#np.random.rand(1))
U2 = y - lamda * U

#U2 = K.concatenate([U, 1-U], axis = -1)

U2 = K.sigmoid(U2)


#loss = K.binary_crossentropy(U2, label)
loss = K.square(U2 - label)
loss = K.mean(loss)

grad = K.gradients(loss, [W,b,lamda])


step = 0.05
iterate = K.function([x, y,label], grad)#, updates=[(W,W-step*grad[0]), (b,b-step*grad[1]), (lamda, lamda-step*grad[2])])


    
#    return iterate

def getData(Set):
    X = np.zeros((len(Set), 1))
    Y = np.zeros((len(Set), 1))
    L = np.zeros((len(Set), 1))
    for i in range(len(Set)):
        X[i] = Set[i][0]
        Y[i] = Set[i][1]
        L[i] = Set[i][2]
    
    return [X, Y, L]

trainSet=[(-0.5,12,0),(0.5,13.2,1),(0.8,8,0),(1,9,1),(1.3,6.5,1),(1.5,5,1),(1.7,3,0),(2.0,1.5,0),(2.2,2,1)
,(2.5,1,1),(3,-1.3,0),(3.3,0.5,1),(3.5,1.5,1),(3.8,1.1,0),(4.1,2.8,1),(4.5,5.1,1),(4.9,6.8,0)
,(5.3,11.2,1),(5.7,14.1,0),(6.2,21.2,1)]

testSet=[(0.7,11.1,1),(1.3,5.2,0),(1.7,3.0,0),(2.2,1.5,1),(2.6,0.5,1)
,(3.2,-0.7,0),(3.7,1.3,1),(4.3,3.5,1),(4.9,6.8,0),(5.5,13.2,1)]

X_train, Y_train, L_train = getData(trainSet)
X_test, Y_test, L_test = getData(testSet)


#iterate = Solve()

for i in range(1):
    grad = iterate([X_train, Y_train, L_train])
    pred = Y_train - K.eval(lamda)*(K.eval(W)*X_train + K.eval(b))
    
    print pred[0:10,],'\n'
    print grad

#inp = np.array([-0.5,12])
#pred = inp[1] - K.eval(lamda)*(K.eval(W)*inp[0] + K.eval(b))
#print pred

