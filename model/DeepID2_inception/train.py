#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:27:52 2017

@author: hh
"""

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('/home/hh/Documents/caffe/models/DeepID2_inception/solver.prototxt')
#weight = '/home/hh/Documents/caffe/models/Inception/bvlc_googlenet_quick_iter_40000.caffemodel'
#solver.net.copy_from(weight)
solver.solve()

