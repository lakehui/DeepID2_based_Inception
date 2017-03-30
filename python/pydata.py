import caffe

import numpy as np
from PIL import Image, ImageOps

import random

class VeriDataLayer(caffe.Layer):
    #""" Load (image1,image2, idlabel, verilabel) """"

    def setup(self, bottom, top):
        #""" Setup data layer according to parameters """
        # config
        params = eval(self.param_str)
        # Check the parameters for validity.
        #check_params(params)
        
        #self.voc_dir = params['voc_dir']
        
        self.mean = np.array(params['mean'])
        self.batch_size = params['batch_size']
        self.source = params['source']
        #self.split = params['split']
        self.width = params['width']
        self.height = params['height']
        self.crop = params['crop']
        self.mirror = params['mirror']
        
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 5:
            raise Exception("Need to define five tops: data1,data2 and label1,lable2,label_veri.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        #split_f  = '{}.txt'.format(self.source)
        #print split_f
        self.indices = open(self.source, 'r').read().splitlines()
        self.idx = range(len(self.indices))
        self.cur_line = 0
        
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.batch_size, 3, self.width, self.height)
        top[1].reshape(self.batch_size, 3, self.width, self.height)        
        # Note the 20 channels (because PASCAL has 20 classes.)
        top[2].reshape(self.batch_size)
        top[3].reshape(self.batch_size)
        top[4].reshape(self.batch_size)
        
        #top[0].reshape(1, *self.data_split1.shape)
        #top[1].reshape(1, *self.data_split2.shape)
        #top[2].reshape(1, *self.label_split1.shape)
        #top[3].reshape(1, *self.label_split2.shape)

        # make eval deterministic
        #if 'train' not in self.split:
        #   self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            random.shuffle(self.idx)


    def reshape(self, bottom, top):
        pass


    def forward(self, bottom, top):
        # load image + label image pair
        batch_half = self.batch_size
        idx_batch1 = self.idx[self.cur_line : self.cur_line+batch_half]
        #get the batch2 according to idx_batch1
        idx_batch2 = []
        for i in range(batch_half):
            tmp_x = idx_batch1[i] + random.randint(1,80)
            if tmp_x >= len(self.indices):
                tmp_x = len(self.indices)- 1
            idx_batch2.append(tmp_x)
        #print 'batch size: ', len(idx_batch1), len(idx_batch2)
        
        self.data_split1, self.label_split1 = self.load_data(idx_batch1)
        self.data_split2, self.label_split2  = self.load_data(idx_batch2)
        self.label_veri = [int(x==y) for x, y in zip(self.label_split1, self.label_split2)]
        #self.label_split1 = self.load_label(idx_batch1)
        #self.label_split2 = self.load_label(idx_batch2)

        # assign output
        top[0].data[...] = self.data_split1
        top[1].data[...] = self.data_split2
        top[2].data[...] = self.label_split1
        top[3].data[...] = self.label_split2
        top[4].data[...] = self.label_veri

        # pick next input
        self.cur_line += batch_half
        if self.cur_line + batch_half >= len(self.indices):
            self.cur_line = 0
            if self.random:
                random.seed(self.seed)
                random.shuffle(self.idx)


    def backward(self, top, propagate_down, bottom):
        pass


    def load_data(self, idx_batch):
        #""" Load input image and preprocess for Caffe: - cast to float - switch channels RGB -> BGR - subtract mean - transpose to channel x height x width order """
        ins_ = np.zeros((len(idx_batch), 3, self.width, self.height))
        labels = np.zeros((len(idx_batch)))
        tt = 0
        for idx in idx_batch:
            path_img, label = self.indices[idx].split(' ')
            im = Image.open(path_img)
            im = im.convert(mode= 'RGB')
            #if mirror is true, random mirror image
            if self.mirror:
                if random.randint(0,1):
                    im = ImageOps.mirror(im)
            #if the crop is true, random crop image
            if self.crop:
                h = im.height; w = im.width
                im = ImageOps.crop(im, (random.randint(1, w/8), random.randint(1, h/8), random.randint(1, w/8),
                                        random.randint(1, h/8)))
            #print 'image size: ', im.size
            
            im = im.resize((self.width, self.height))
            in_ = np.array(im, dtype=np.float32)
            #in_ = in_[:,:,::-1]
            in_ -= self.mean
            in_ = in_.transpose((2,0,1))
            ins_[tt, ...] = in_
            labels[tt, ...] = int(label)
            tt += 1
        #print 'the top size is :', ins_.shape, labels.shape
        return ins_, labels


#    def load_label(self, idx_batch):
#        """ Load label image as 1 x height x width integer array of label indices. The leading singleton dimension is required by the loss. """
#        im = Image.open('{}/SegmentationClass/{}.png'.format(self.voc_dir, idx))
#        label = np.array(im, dtype=np.uint8)
#        label = label[np.newaxis, ...]
#        return label
