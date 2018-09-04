#!usr/bin/env python
#-*-coding:utf-8-*-

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets  
import torchvision.transforms as transforms
import numpy as np
import caffe
from torch.autograd import Variable
from PIL import Image
import glob

import torch.utils.model_zoo as model_zoo
import math
import os
import sys
#sys.path.append('/home/meitu/caffeserver_xcb/caffe/python/')
os.environ["GLOG_minloglevel"] = "3"
import caffe
global log_flag


def test_pytorch():
    file_path,a,file_py = sys.argv[1].rpartition('/')
    file_name,b,py = file_py.rpartition('.')
    print file_path, a, file_py, file_name

    sys.path.append(os.path.dirname(file_path+'/'+file_name+'.py'))
    model_file = __import__(file_name)
    model,images = model_file.get_model_and_input()
    model.eval()
    print("==model input===============")
    # do test
    print model,images
    print "==output==============="
    out = model(images)


    for v in out:
        print '---------------------'
        print type(v)
        v = v.data.numpy().flatten()
        print v[:10]
        i_min = v.argmin()
        i_max = v.argmax()
        print("max index :",i_max)
        print("max number :",v[i_max])
        print("min index :",i_min)
        print("min number :",v[i_min])
    print("end")




def test_caffe(input_image):
    file_path,a,file_py = sys.argv[1].rpartition('/')
    file_name,b,py = file_py.rpartition('.')
    MODEL_FILE = file_path+'/'+file_name+'.prototxt' 
    PRETRAINED = file_path+'/'+file_name+'.caffemodel' 

    caffe.set_mode_cpu()
    net = caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)

    if log_flag == True:
        print dir(net.blobs)
        print (net.blobs.keys())
        print dir(net.blobs['data'])
        print type(net.blobs['data'].data)
        print type(net.blobs['data'].data.shape)
    #input_image = np.ones(net.blobs['data'].data.shape)
    net.blobs['data'].data[...] = input_image
    out = net.forward()
    print "num of output:", len(out)
    lastlayer_name =net.blobs.keys()[-1]
    if log_flag == True:
        print "==input==============="
        print input_image
        print input_image.shape
        print "==output==============="
        print out
        print ("The last layer: ",last_layer)


    results = {}
    results[lastlayer_name] = net.blobs[lastlayer_name].data
    print "last:",lastlayer_name
    if log_flag == True:
        print results[lastlayer_name]
        print dir(results[lastlayer_name])
        print results[lastlayer_name].shape
    for k in results:
        v = results[k].flatten()
        i_min = v.argmin()
        i_max = v.argmax()
        if log_flag == True:
            print '---------------------'
            print  "The last layer name: ",k
            print ("the pre ten number: ",v[:10])
            print("max index :",i_max)
            print("max number :",v[i_max])
            print("min index :",i_min)
            print("min number :",v[i_min])
            print("the caffe test  end")
            print '---------------------'

    if log_flag == True :
        print net.blobs.keys()
        print 
        print (net.params.keys())
        print (net.params.items())
        print (net.params.values())
        #print (net.params.values().data[...])
        def printParams(vec):
            for v in vec:
                print v.data
        for x in net.params.keys():
            print x
            if len(net.params[x]) == 1:
                out_weight1 = np.asarray(net.params[x][0].data[...])
                print out_weight1
                out_weight = np.average(np.absolute(out_weight1))
                print "weight:",out_weight
            elif len(net.params[x]) == 2:
                out_weight1 = np.asarray(net.params[x][0].data[...])
                out_weight = np.average(np.absolute(out_weight1))
                out_bias1 = np.asarray(net.params[x][1].data[...])
                out_bias  = np.average(np.absolute(out_bias1))
                print "weight:",out_weight
                print "bias:",out_bias
            else:
                out_weight1 = np.asarray(net.params[x][0].data[...])
                out_weight = np.average(np.absolute(out_weight1))
                out_bias1 = np.asarray(net.params[x][1].data[...])
                out_bias  = np.average(np.absolute(out_bias1))
                out_temp1 = np.asarray(net.params[x][2].data[...])
                out_temp  = np.average(np.absolute(out_temp1))
                print "weight:",out_weight
                print "bias:",out_bias
                print "temp:",out_temp


        '''
        #printParams(net.params['scale1'])
        #print (net.params['BatchNorm1'][0].data)
        print "total layers:", len(net.layers)
        print dir(net.layers[3])
        print (net.layers[3].type)
        print len(net.layers[4].blobs)
        print "x"
        printParams(net.layers[3].blobs)

        print dir(net)
        print dir(net.layers[3])
        print net.layers[3].__class__
        '''
    return results[lastlayer_name],i_max,i_min
