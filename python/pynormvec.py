#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:13:12 2017

@author: hh

norm vector layder of caffe
"""

import caffe

import numpy as np
from PIL import Image, ImageOps
import copy



class NormVecLayer(caffe.Layer):

    def setup(self, bottom, top):
        #""" Setup data layer according to parameters """
        # config
#        print "enter normveclayer"
        #params = eval(self.param_str)
        #Check the parameters for validity.
        #self.check_params(params)

        # one tops: feature vector
        if len(top) != 1:
            raise Exception("Only need to define one tops: feature vector.")
        # data layers have no bottoms
        if len(bottom) != 1:
            raise Exception("not define bottom.")

        self.bottom_data = np.zeros(bottom[0].data.shape)
        
        self.bottom_norm = np.zeros(bottom[0].data.shape[0])
        # reshape tops to fit (leading 1 is for batch dimension)
        
        
#        print top[0].data.shape


    def reshape(self, bottom, top):
        #print bottom[0].data.shape
        if len(bottom[0].data.shape) == 2:
            top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1])
        else:
            top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1], 1, 1)
        #print top[0].data.shape
        pass
           


    def forward(self, bottom, top):
        self.bottom_data = copy.deepcopy(bottom[0].data)
        
        bottom_copy = bottom[0].data
        for i in range(bottom_copy.shape[0]):           
            f_vec = bottom_copy[i,].reshape(bottom_copy.shape[1])
            self.bottom_norm[i] = np.sqrt(float(f_vec.dot(f_vec)))
            top[0].data[i,] = bottom_copy[i,]/self.bottom_norm[i]
        
        #print "self", self.bottom_data
        

    def backward(self, top, propagate_down, bottom):
        #print "top back", top[0].diff[0,]
        self.top_diff = copy.deepcopy(top[0].diff)
        if propagate_down[0]:
            for i in range(bottom[0].data.shape[0]):
                inter_diff = -1.0*np.square(self.bottom_data[i,])/np.power(self.bottom_norm[i], 3) + 1.0/self.bottom_norm[i]
                bottom[0].diff[i,] = top[0].diff[i,]*inter_diff
        #print "bottom back:", bottom[0].diff[0,]
        


    def check_params(params):
        if len(params) != 0:
            raise Exception("don't define any params")
