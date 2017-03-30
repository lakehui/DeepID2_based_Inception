#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 00:09:01 2017

@author: hh
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:44:05 2017

@author: hh
"""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()  
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config= tf_config))


from keras import backend as K
import numpy as np


def getData(Set):
    X = np.zeros((len(Set), 1))
    Y = np.zeros((len(Set), 1))
    L = np.zeros((len(Set), 1))
    for i in range(len(Set)):
        X[i] = Set[i][0]
        Y[i] = Set[i][1]
        L[i] = Set[i][2]
    
    return [X, Y, L]

trainSet=[(-0.5,12,-1),(0.5,13.2,1),(0.8,8,-1),(1,9,1),(1.3,6.5,1),(1.5,5,1),(1.7,3,-1),(2.0,1.5,-1),(2.2,2,1)
,(2.5,1,1),(3,-1.3,-1),(3.3,0.5,1),(3.5,1.5,1),(3.8,1.1,-1),(4.1,2.8,1),(4.5,5.1,1),(4.9,6.8,-1)
,(5.3,11.2,1),(5.7,14.1,-1),(6.2,21.2,1)]

testSet=[(0.7,11.1,1),(1.3,5.2,-1),(1.7,3.0,-1),(2.2,1.5,1),(2.6,0.5,1)
,(3.2,-0.7,-1),(3.7,1.3,1),(4.3,3.5,1),(4.9,6.8,-1),(5.5,13.2,1)]

X_train, Y_train, L_train = getData(trainSet)

X_test, Y_test, L_test = getData(testSet)


#--------------buile graph model----------------------------
grad_t=[0,0,0]

K.set_epsilon(1e-11)
x = K.placeholder(shape=(None, 1), dtype= 'float32')
y = K.placeholder(shape=(None, 1), dtype= 'float32')
label = K.placeholder(shape=(None, 1), dtype= 'float32')

#x= K.variable(value = X_train)
#y = K.variable(value = Y_train)
#label = K.variable(value = L_train)

W = K.variable(value = 1.)#np.random.rand(1))
b = K.variable(value = -1.)#np.random.rand(1))
U = K.square(W*x + b)


lamda = K.variable(value = 1.)#np.random.rand(1))
U2 = y - lamda * U

U2 = K.tanh(U2)

loss = K.square(U2 - label)
loss = K.mean(loss)

grad = K.gradients(loss, [W,b,lamda])


#--------------------uniformize gradient
sum_grad = 0
for i in range(len(grad)):
    sum_grad += K.square(grad[i])
    
for i in range(len(grad)):
    grad[i] /= (K.sqrt(sum_grad) + 1e-5)

#--------using sgd with momentum     
momentum = 0.7
for i in range(len(grad)):
    grad[i] = grad_t[i]* momentum + grad[i]
    

step = K.variable(value=0.05)
iterate = K.function([x,y,label], [U2], updates=[(W,W-step*grad[0]), (b,b-2*step*grad[1]), (lamda, lamda-step*grad[2])])


#-------------------solve----------------------------------------
for i in range(200):
    pred = iterate([X_train, Y_train, L_train])
    if i >100:
        K.set_value(step, 0.001)
    
    pred_label = np.sign(pred)
#    pred = Y_train - K.get_value(lamda)*(K.get_value(W)*X_train + K.get_value(b))*(K.get_value(W)*X_train + K.get_value(b))
#    preds = K.get_value(K.sign(pred))
    acc = 1- np.mean(np.abs(pred_label - L_train)/2.)
    
    print 'acc_train: ', acc


#-------------------test----------------------------------------
pred = Y_test - K.eval(lamda)*(K.eval(W)*X_test + K.eval(b))*(K.eval(W)*X_test + K.eval(b))
preds = K.eval(K.sign(pred))
print 'predict result: ', preds
acc_test = 1- np.mean(np.abs((preds - L_test))/2.)
print 'acc_test: ', acc_test
