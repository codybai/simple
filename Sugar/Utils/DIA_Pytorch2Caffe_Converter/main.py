#!usr/bin/env python
#-*-coding:utf-8-*-

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import glob
import numpy as np
np.set_printoptions(precision = 8)
torch.set_printoptions(precision = 8)
from termcolor import colored
import os
os.environ["GLOG_minloglevel"] = "3"
import argparse
import sys
import caffe

caffe.set_mode_cpu()
from caffe import layers as L,params as P,to_proto

from caffe.proto import caffe_pb2
from google.protobuf import text_format
import torch.utils.model_zoo as model_zoo
import math
import inspect
class ToCaffe():
    def __init__(self):
        self.l_bottom = 0
        self.l_top = 0
        self.l_top1 = 0
        self.l_data = 0
        self.output = 0
        self.target_dir = 0
        self.i1 = 0
        self.i2 = 0
        self.i3 = 0
        self.i4 = 0
        self.i5 = 0
        self.j = 0
        self.k = 0
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.l = 0
        self.p = 0
        self.p1 = 0
        self.q = 0
        self.o = 0
        self.o2 = 0
        self.o3 = 0
        self.o4 = 0
        self.o5 = 0
        self.o_s = 0
        self.o_max = 0
        self.r = 0
        self.r1 = 0
        self.frame_v = [[], []]
        self.frame_match = [[], [], []]
        self.py_params = {}
        self.target_dir = ''
        self.log_flag = False
    def getbottom(self,v, layer_name=""):
        found_idx = None
        for i, item in enumerate(self.frame_match[2]):
            if item == id(v):
                if self.log_flag == True:
                    print("i:", i)
                    print(self.frame_match[1][i])
                found_idx = i
                # break
        if self.log_flag == True:
            # print(frame_match)
            print("id :", self.frame_match[2][found_idx])

        if found_idx is None:
            print colored('[error not found bottom]', 'red'), colored('%s' % (layer_name), 'blue')
            sys.exit(-2)
        else:
            return self.frame_match[0][found_idx]
    def put_frame(self,name,top,arg):
        self.frame_v[0].append(name)
        self.frame_v[1].append(top)
        self.frame_match[0].append(top)
        self.frame_match[1].append(arg)
        self.frame_match[2].append(id(arg))
    def message(self,msg1, msg2, status='ok'):
        print colored('[%s]' % (status), 'green'), colored('%s-->%s' % (msg1, msg2), 'blue')

    def caller_name(self,skip=2):
        """Get a name of a caller in the format module.class.method

           `skip` specifies how many levels of stack to skip while getting caller
           name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

           An empty string is returned if skipped levels exceed stack height
        """
        stack = inspect.stack()
        start = 0 + skip
        if len(stack) < start + 1:
            return ''
        parentframe = stack[start][0]

        name = []
        module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        # TODO(techtonik): consider using __main__
        if module:
            name.append(module.__name__)
        # detect classname
        if 'self' in parentframe.f_locals:
            # I don't know any way to detect call from the object method
            # XXX: there seems to be no way to detect static method call - it will
            #      be just a function call
            name.append(parentframe.f_locals['self'].__class__.__name__)
        codename = parentframe.f_code.co_name
        if codename != '<module>':  # top level usually
            name.append(codename)  # function or a method
        del parentframe
        return ".".join(name)

    def handel(self,layer, layer_input, arg, frame):
        if type(layer) is nn.Linear:
            self.l += 1
            self.name = 'ip' + `self.l`
            if self.log_flag == True:
                print("-------------------converting {}: {}".format(self.name, type(layer)))
                print("linear:", layer.in_features, layer.out_features)
                print("weight size, bias size:", layer.weight.size(), layer.bias.size())
            type_name = "InnerProduct"
            if layer.bias is not None:
                self.py_params[self.name] = (layer.weight, layer.bias)
            else:
                self.py_params[self.name] = (layer.weight,)

            self.l_bottom = self.getbottom(layer_input[0], str(layer))
            self.l_top = L.InnerProduct(self.l_bottom, num_output=layer.out_features, name=name)
            self.put_frame(self.name, self.l_top, arg)
            self.message("nn.Linear", type_name, 'ok')
        elif type(layer) is nn.Conv2d:
            pass
        elif type(layer) is nn.AvgPool2d:
            pass
        elif type(layer) is nn.MaxPool2d:
            pass
        elif type(layer) is nn.Sequential:
            pass
        elif type(layer) is nn.Dropout2d:
            self.o3 += 1
            self.name = "drop" + `self.o3`
            self.frame_match[0].append(self.l_top)
            self.frame_match[1].append(arg)
            self.frame_match[2].append(id(arg))
            self.l_bottom = self.getbottom(layer_input[0], str(layer))
            self.message("nn.Dropout2d", "none", 'skip')
        elif type(layer) is nn.ConvTranspose2d:
            pass
        elif type(layer) is nn.UpsamplingNearest2d:
            self.i3 += 1
            name = "deconv" + `self.i3`
            if self.log_flag == True:
                print("-------------------converting {}: {}".format(name, type(layer)))
                print("UpsamplingNearest2dxx:", layer_input[0].size()[1], layer.scale_factor)
                print("-----------------hah---------")
                print(type(layer_input[0].size()[1]))
                print(dir(layer.scale_factor))
                print("--------------------hah------")
            type_name = "Deconvolution"

            # num_output = layer.size
            num_output = layer_input[0].size()[1]
            kernel = layer.scale_factor
            stride = layer.scale_factor

            weight = Variable(torch.Tensor(num_output, 1, kernel, kernel).fill_(1))
            # print(weight.data)
            self.py_params[name] = (weight,)

            self.l_bottom = self.getbottom(layer_input[0], str(layer))
            # l_top = L.Deconvolution(l_bottom, name=name, convolution_param=dict(num_output=num_output, kernel=kernel, stride=stride))
            self.l_top = L.Deconvolution(self.l_bottom, name=name,
                                    convolution_param=dict(num_output=num_output, group=num_output, kernel_size=kernel,
                                                           stride=stride))

            self.put_frame(self.name, self.l_top, arg)
            self.message("nn.UpsamplingNearest2d", type_name, 'ok')

        elif type(layer) is nn.BatchNorm2d:
            self.p1 += 1
            self.name = 'BatchNorm' + `self.p1`
            if self.log_flag == True:
                print("-------------------converting {}: {}".format(self.name, type(layer)))
                print("type  mean ", type(layer.running_mean))
                print(
                "bn2d mean var  weight bias  training affine momentum epsxx:", layer.running_mean, layer.running_var,
                layer.weight, layer.bias, layer.training, layer.affine, layer.momentum, layer.eps)
                print("id====== :", id(layer_input))
                print("variable====== :", layer_input)
            type_name = 'BatchNorm'

            self.l_bottom = self.getbottom(layer_input[0])
            # l_top1 = L.BatchNorm(l_bottom, name=name, eps=layer.eps, use_global_stats = True)
            self.l_top1 = L.BatchNorm(self.l_bottom, name=self.name, eps=layer.eps, use_global_stats=False)
            self.py_params[self.name] = (layer.running_mean, layer.running_var)
            self.put_frame(self.name, self.l_top1, arg)

            self.name1 = "scale" + `self.p1`
            type_name1 = 'Scale'

            self.l_bottom1 = self.l_top1
            self.l_top = L.Scale(self.l_bottom1, name=self.name1, bias_term=True)
            self.py_params[self.name1] = (layer.weight, layer.bias)
            self.put_frame(self.name1, self.l_top, arg)
            self.message("nn.BatchNorm2d", "BatchNorm + Scale", 'ok')
            # raw_input()
        elif type(layer) is nn.ReLU:
            self.k += 1
            self.name = 'relu' + `self.k`
            if self.log_flag == True:
                print("-------------------converting {}: {}".format(self.name, type(layer)))
            type_name = 'ReLU'
            self.l_bottom = self.getbottom(layer_input[0], str(layer))
            self.l_top = L.ReLU(self.l_bottom, name=self.name, in_place=True)

            self.put_frame(self.name, self.l_top, arg)
            self.message("nn.ReLU", type_name, 'ok')
        elif type(layer) is nn.PReLU:
            self.k1 += 1
            self.name = 'prelu' + `self.k1`
            type_name = 'PReLU'
            if self.log_flag == True:
                print("-------------------converting {}: {}".format(self.name, type(layer)))
            self.l_bottom = self.getbottom(layer_input[0], str(layer))
            self.l_top = L.PReLU(self.l_bottom, name=self.name)
            self.py_params[self.name] = (layer.weight,)
            self.put_frame(self.name, self.l_top, arg)
            self.message("nn.PReLU", type_name, 'ok')
        elif type(layer) is nn.LeakyReLU:
            self.k1 += 1
            self.name = 'relu' + `self.k1`
            type_name = 'ReLU'
            if self.log_flag == True:
                print("-------------------converting {}: {}".format(self.name, type(layer)))

            self.l_bottom = self.getbottom(layer_input[0], str(layer))
            self.l_top = L.ReLU(self.l_bottom, name=self.name, relu_param=dict(negative_slope=layer.negative_slope))
            self.put_frame(self.name, self.l_top, arg)
            self.message("nn.LeakyPReLU", type_name, 'ok')
        elif type(layer) is nn.Tanh:
            self.k2 += 1
            name = 'tanh' + `self.k2`
            type_name = 'Tanh'
            if self.log_flag == True:
                print("-------------------converting {}: {}".format(name, type(layer)))

            self.l_bottom = self.getbottom(layer_input[0], str(layer))
            self.l_top = L.TanH(self.l_bottom, name=name)
            self.put_frame(self.name, self.l_top, arg)
            self.message("nn.Tanh", type_name, 'ok')
        elif os.path.dirname(frame.f_back.f_code.co_filename) == self.target_dir:
            # print (frame.f_locals.keys())
            l_output = frame.f_locals["result"]
            if type(l_output) is tuple:
                v = l_output[0]
            else:
                v = l_output
            self.l_v = self.getbottom(v, str(layer))

    def traceit(self,frame, event, arg):
        # print event
        # print caller_name()
        if event == 'return' and frame.f_back and frame.f_code.co_name != "__getattr__" and frame.f_code.co_name != "mark_shared_storage" and (
                        self.caller_name().find("torch.nn") == 0 or self.caller_name().find(
                "torch.autograd") == 0) and frame.f_code.co_name != "__repr__":
            # print "%s at %s lineno:%s" %( frame.f_code.co_name, frame.f_code.co_filename ,frame.f_code.co_firstlineno)
            # print "fram fileno:",frame.f_lineno
            # print "caller :",caller_name()
            if self.log_flag == True:
                print("=================== %s ==============================================" % frame.f_code.co_name)
                print('=当前层 locals keys--')
                print(frame.f_locals.keys())  # 当前层的函数参数
                print(frame.f_locals)
                print("--当前层的 varnames----")
                print(frame.f_code.co_varnames)  # 输出当前层的变量名
                for var in frame.f_code.co_varnames:
                    if var in frame.f_locals:
                        print("--当前层的 var, value----")
                        v = frame.f_locals[var]
                        if type(v) is not Variable:
                            # print(var, ':', v)
                            pass
                        else:
                            print(var, ': pytorch variab')
                print(frame.f_back.f_code.co_name)  # 当前层的上一层的名字
                # print(dir(frame))
                print("===&&&frame.f_code.co_name &&&&&&&&&&&&&&&&&&&&&&&&&&===")
                print('frame name :', frame.f_code.co_name)
                # print('frame.getattribute :',frame.__getattribute__)
                # print('frame.iadd :',frame.__iadd__)
                print("===&&&&&&&&&&&&&&&&&&&&&&&&&&&&&===")
                print(event, frame.f_code.co_name, frame.f_code, frame, frame.f_back)  # 事件  当前层的名称   当前层的对象位置  上一层的对象位置

            if frame.f_code.co_name == "__call__":
                m = frame.f_locals["self"]  # 作用域下的参数
                l_input = frame.f_locals["input"]
                if self.log_flag == True:
                    # print("input:",l_input)
                    print ('m:', m, m.state_dict().keys())
                    print('frame class :', frame.f_locals["self"].__class__)
                    print('frame class name :', frame.f_locals["self"].__class__.__name__)
                    print('frame class dict:', frame.f_locals["self"].__class__.__dict__)
                    print('frame class dir :', dir(frame.f_locals["self"].__class__))
                    # pdb.set_trace()
                    # print ('m.state_param:', m.state_dict().values())
                    # print (arg)
                    print ("handel nn.event :")
                self.handel(m, l_input, arg, frame)
            elif frame.f_code.co_name == "conv2d":
                self.i4 += 1
                name = "conv" + `self.i4`
                type_name = "Convolution"
                groups = frame.f_locals["groups"]
                padding = frame.f_locals["padding"]
                if type(padding) == tuple:
                    pad_h = frame.f_locals["padding"][0]
                    pad_w = frame.f_locals["padding"][1]
                elif type(padding) == int:
                    pad_h = padding
                    pad_w = padding

                stride = frame.f_locals["stride"]
                if type(stride) == tuple:
                    stride_h = frame.f_locals["stride"][0]
                    stride_w = frame.f_locals["stride"][1]
                elif type(stride) == int:
                    stride_h = stride
                    stride_w = stride
                dilation = frame.f_locals["dilation"]
                if type(dilation) == tuple:
                    dilation = frame.f_locals["dilation"][0]
                weight = frame.f_locals["weight"]
                bias = frame.f_locals["bias"]
                num_output = weight.size()[0]
                kernel_h = weight.size()[-2]
                kernel_w = weight.size()[-1]

                if self.log_flag == True:
                    print "pad:", pad_h, pad_w
                    print "stride:", stride_h, stride_w
                    print type(dilation)
                    print "dilation", dilation
                    print weight.size()
                    print "====groups:", groups
                    print "====num_output:", num_output
                    print "kernel_h:", kernel_h
                    print "kernel_w:", kernel_w

                l_input = frame.f_locals["input"]
                l_bottom = self.getbottom(l_input, type_name)
                if bias is not None:
                    self.py_params[name] = (weight, bias)
                else:
                    self.py_params[name] = (weight,)

                if self.l_bottom == self.l_data:
                    self.l_top = L.Convolution(name=name, bottom="data",
                                          kernel_h=kernel_h, kernel_w=kernel_w,
                                          num_output=num_output,
                                          group=groups, stride_h=stride_h, stride_w=stride_w,
                                          pad_h=pad_h, pad_w=pad_w,
                                          dilation=dilation)
                else:
                    self.l_top = L.Convolution(l_bottom, name=name, kernel_h=kernel_h, kernel_w=kernel_w,
                                          num_output=num_output, group=groups, stride_h=stride_h, stride_w=stride_w,
                                          pad_h=pad_h, pad_w=pad_w, dilation=dilation)

                self.put_frame(name, self.l_top, arg)
                self.message("Conv2d", type_name, 'ok')
                # raw_input()
            elif frame.f_code.co_name == "conv_transpose2d":
                self.i5 += 1
                name = "deconv" + `self.i5`
                type_name = "Deconvolution"

                output_padding = frame.f_locals["output_padding"]
                if type(output_padding) == tuple:
                    if output_padding[0] != 0 or output_padding[1] != 0:
                        print colored('[Not support]', 'red'), colored(
                            "output_padding param is not exist in caffe Deconvolution yet.", 'blue')
                        sys.exit(-1)
                elif type(output_padding) == int:
                    if output_padding != 0:
                        print colored('[Not support]', 'red'), colored(
                            "output_padding param is not exist in caffe Deconvolution yet.", 'blue')
                        sys.exit(-1)

                groups = frame.f_locals["groups"]
                padding = frame.f_locals["padding"]
                if type(padding) == tuple:
                    pad_h = frame.f_locals["padding"][0]
                    pad_w = frame.f_locals["padding"][1]
                elif type(padding) == int:
                    pad_h = padding
                    pad_w = padding
                stride = frame.f_locals["stride"]
                if type(stride) == tuple:
                    stride_h = frame.f_locals["stride"][0]
                    stride_w = frame.f_locals["stride"][1]
                elif type(stride) == int:
                    stride_h = stride_w = stride
                weight = frame.f_locals["weight"]
                bias = frame.f_locals["bias"]
                num_output = weight.size()[1] * groups
                kernel_h = weight.size()[-2]
                kernel_w = weight.size()[-1]

                if self.log_flag == True:
                    print output_padding
                    print type(output_padding)
                    print "pad_h,pad_w:", pad_h, pad_w
                    print type(stride)
                    print stride_h, stride_w
                    print "weight", weight
                    print "bias", bias
                    print weight.size()
                    print "name:", name
                    print "===groups:", groups
                    print "===num_output:", num_output
                    print "kernel_h", kernel_h
                    print "kernel_w", kernel_w

                if bias is not None:
                    self.py_params[name] = (weight, bias)
                else:
                    self.py_params[name] = (weight,)
                l_input = frame.f_locals["input"]
                l_bottom = self.getbottom(l_input, type_name)
                if self.log_flag == True:
                    print "input size:", l_input.size()
                    print "arg size:", arg.size()
                l_top = L.Deconvolution(l_bottom, name=name,
                                        convolution_param=dict(num_output=num_output, kernel_h=kernel_h,
                                                               kernel_w=kernel_w, stride_h=stride_h, stride_w=stride_w,
                                                               pad_h=pad_h, pad_w=pad_w, group=groups))

                self.self.put_frame(name, l_top, arg)
                self.message("Contranspose2d", type_name, 'ok')
                # raw_input()
            elif frame.f_code.co_name == "avg_pool2d":
                self.o += 1
                name = "avgpool" + `self.o`
                if self.log_flag == True:
                    print("-------------------converting {}: {}".format(name, frame.f_code.co_name))
                type_name = "MTPooling"
                kernel_size1 = frame.f_locals["kernel_size"]
                pad1 = frame.f_locals["padding"]
                stride1 = frame.f_locals["stride"]
                if stride1 == None:
                    stride1 = kernel_size1
                pool1 = P.Pooling.AVE
                l_input = frame.f_locals["input"]
                l_bottom = self.getbottom(l_input, type_name)
                l_top = L.MTPooling(l_bottom, name=name,
                                    pooling_param=dict(pool=pool1, kernel_size=kernel_size1, stride=stride1, pad=pad1))

                self.self.put_frame(name, l_top, arg)
                self.message(frame.f_code.co_name, type_name, 'ok')
            elif frame.f_code.co_name == "__mul__":
                self.o5 += 1
                name = "scale" + `self.o5`
                if self.log_flag == True:
                    print("-------------------converting {}: {}".format(name, frame.f_code.co_name))
                type_name = "Scale"
                l_input = frame.f_locals["self"]
                other = frame.f_locals["other"]
                l_bottom = self.getbottom(l_input, type_name)
                l_top = L.Scale(l_bottom, name=name, bias_term=False)
                self.py_params[name] = (other,)
                self.self.put_frame(name, l_top, arg)
                self.message(frame.f_code.co_name, type_name, 'ok')
            elif frame.f_code.co_name == "mul":
                self.o5 += 1
                name = "scale" + `self.o5`
                if self.log_flag == True:
                    print("-------------------converting {}: {}".format(name, frame.f_code.co_name))
                type_name = "Scale"
                l_input = frame.f_locals["self"]
                other = frame.f_locals["other"]
                l_bottom = self.getbottom(l_input, type_name)
                l_top = L.Scale(l_bottom, name=name, bias_term=False)
                self.py_params[name] = (other,)
                self.self.put_frame(name, l_top, arg)
                self.message(frame.f_code.co_name, type_name, 'ok')
            elif frame.f_code.co_name == "max_unpool2d":
                self.o2 += 1
                name = "upsample" + `self.o2`
                type_name = "MTPooling"
                kernel_size1 = frame.f_locals["kernel_size"][0]
                pad1 = frame.f_locals["padding"]
                stride1 = frame.f_locals["stride"][0]
                indices = frame.f_locals["indices"]
                output_size = frame.f_locals["output_size"]
                if self.log_flag == True:
                    print("-------------------converting {}: {}".format(name, frame.f_code.co_name))
                    print("indices", indices)
                    print("output_size", output_size[0])
                    print("kernel_size1", kernel_size1)
                    print("stride1", stride1)
                    print("pad1", pad1)
                    print("layer dir", dir(L))
                    print("layer attribute", L.__getattribute__)

                pool1 = P.Pooling.MAX
                l_input = frame.f_locals["input"]
                n = frame.f_locals["indices"]  # 作用域下的参数
                l_bottom = self.getbottom(l_input, type_name)
                l_bottom1 = self.getbottom(n, type_name)

                l_top = L.Upsample(l_bottom, l_bottom1, upsample_h=output_size[0], upsample_w=output_size[1], name=name)

                self.self.put_frame(name, l_top, arg)
                self.message(frame.f_code.co_name, type_name, 'ok')

            elif frame.f_code.co_name == "view":
                if self.log_flag == True:
                    print("-------------------convert {}: {}".format("view", frame.f_code.co_name))
                self.k3 += 1
                name = 'reshape' + `self.k3`
                type_name = 'Reshape'
                sizes = frame.f_locals["sizes"]
                list_dim = []
                for i in range(len(sizes)):
                    list_dim.append(sizes[i])
                l_input = frame.f_locals["self"]
                l_bottom = self.getbottom(l_input, type_name)
                l_top = L.Reshape(l_bottom, name=name, reshape_param={'shape': {'dim': list_dim}})
                self.self.put_frame(name, l_top, arg)
                self.message(frame.f_code.co_name, type_name, 'ok')
            elif frame.f_code.co_name == "repeat":
                type_name = "Repeat"
                self.l_input = frame.f_locals["input"]
                self.l_bottom = self.getbottom(self.l_input, type_name)
                self.frame_match[0].append(self.l_top)
                self.frame_match[1].append(arg)
                self.frame_match[2].append(id(arg))
                self.message(frame.f_code.co_name, type_name, 'ok')
            elif frame.f_code.co_name == "dropout":
                type_name = "None"
                self.l_input = frame.f_locals["input"]
                self.l_bottom = self.getbottom(self.l_input, type_name)
                self.frame_match[0].append(self.l_top)
                self.frame_match[1].append(arg)
                self.frame_match[2].append(id(arg))
                self.message(frame.f_code.co_name, type_name, 'skip')
            elif frame.f_code.co_name == "relu":
                self.k += 1
                self.name = 'relu' + `self.k`
                type_name = 'ReLU'
                l_input = frame.f_locals["input"]
                if self.log_flag == True:
                    print("-------------------converting {}: {}".format(self.name, frame.f_code.co_name))
                    print("id====== :", id(l_input))
                    print("variable====== :", l_input)
                self.l_bottom = self.getbottom(l_input, type_name)
                self.l_top = L.ReLU(self.l_bottom, name=self.name, in_place=True)

                self.self.put_frame(self.name, self.l_top, arg)
                self.message(frame.f_code.co_name, type_name, 'ok')
            elif frame.f_code.co_name == "prelu":
                pass
            elif frame.f_code.co_name == "softmax":
                name = 'prob'
                if self.log_flag == True:
                    print("-------------------converting {}: {}".format(name, frame.f_code.co_name))
                type_name = 'Softmax'
                top = 'prob'
                l_input = frame.f_locals["input"]
                l_bottom = self.getbottom(l_input, type_name)
                l_top = L.Softmax(l_bottom, name=name)

                self.frame_v[0].append(name)
                self.frame_v[1].append(l_top)
                self.frame_match[0].append(l_top)
                self.frame_match[1].append(arg)
                self.message(frame.f_code.co_name, type_name, 'ok')
            elif frame.f_code.co_name == "__iadd__":
                self.q += 1
                self.name = 'eltwise' + `self.q`
                l_input = frame.f_locals["self"]  # 作用域下的参数
                n = frame.f_locals["other"]  # 作用域下的参数
                if self.log_flag == True:
                    print("-------------------converting {}: {}".format(self.name, frame.f_code.co_name))
                    print ('l_input:', l_input)
                    print ('n  top:', n.__class__)
                    print ('n id :', id(n))
                    print ('n type:', type(n))
                    print('frame class :', frame.f_locals["self"].__class__)
                    print("id==l_input==== :", id(l_input))
                    print("Variable==l_input==== :", l_input)
                    print("id===n=== :", id(n))
                    print("Variable==n==== :", n)
                type_name = 'Eltwise'

                l_bottom = self.getbottom(l_input, type_name)
                l_bottom_1 = self.getbottom(n, type_name)
                l_top = L.Eltwise(l_bottom_1, l_bottom, operation=P.Eltwise.SUM, name=self.name)

                self.self.put_frame(self.name, l_top, arg)
                self.message(frame.f_code.co_name, type_name, 'ok')
            elif frame.f_code.co_name == "__add__":
                self.q += 1
                name = 'eltwise' + `self.q`
                l_input = frame.f_locals["self"]  # 作用域下的参数
                n = frame.f_locals["other"]  # 作用域下的参数
                if self.log_flag == True:
                    print("-------------------converting {}: {}".format(name, frame.f_code.co_name))
                    print ('l_input:', l_input)
                    print("=====================================================================")
                    print ('n id :', id(n))
                    print ('n type:', type(n))
                    print('frame class :', frame.f_locals["self"].__class__)
                    print("id==l_input==== :", id(l_input))
                    print("id===n=== :", id(n))
                    print (self.frame_v)
                    print (self.frame_match)
                type_name = 'Eltwise'

                l_bottom = self.getbottom(l_input, type_name)
                l_bottom_1 = self.getbottom(n, type_name)

                l_top = L.Eltwise(l_bottom_1, l_bottom, operation=P.Eltwise.SUM, name=name)

                self.put_frame(name, l_top, arg)
                self.message(frame.f_code.co_name, type_name, 'ok')

            elif frame.f_code.co_name == "cat":
                self.r += 1
                name = 'concat' + `self.r`
                dim = frame.f_locals["dim"]  # 作用域下的参数
                Variables = frame.f_locals["iterable"]  # 作用域下的参数
                if self.log_flag == True:
                    print("-------------------converting {}: {}".format(name, frame.f_code.co_name))
                    print("=====================================================================")
                    print ('dim :', dim)
                    print ('Variables :', Variables)
                    print ('Variables type:', type(Variables))
                type_name = 'Concat'
                l_bottoms = [0 for x in range(len(Variables))]
                for iterator in range(len(Variables)):
                    l_bottoms[iterator] = self.getbottom(Variables[iterator], type_name)
                    if self.log_flag == True:
                        print("id==Variables[iteraor]==== :", id(Variables[iterator]))
                if self.log_flag == True:
                    print(l_bottoms)
                    print("=====================================================================")
                l_top = L.Concat(*l_bottoms, name=name, axis=dim)

                self.put_frame(name, l_top, arg)
                self.message(frame.f_code.co_name, type_name, 'ok')
            elif frame.f_code.co_name == "__getitem__" and frame.f_locals.has_key("key"):
                self.o_s += 1
                name = 'slice' + `self.o_s`
                type_name = 'Slice'
                l_input = frame.f_locals["self"]
                if self.log_flag == True:
                    print("-------------------converting {}: {}".format(name, frame.f_code.co_name))
                    print l_input.size()
                    print l_input.size()[1]
                key = frame.f_locals["key"]

                # make sure other channel has no slice, except channal 1
                for idx in [0, 2, 3]:
                    assert (key[idx].start is None and key[idx].stop is None)

                start = key[1].start
                stop = key[1].stop
                step = key[1].step

                if step is not None:
                    raise RuntimeError("[ERROR]: not support step for slice ")
                if start is None and stop is None:
                    raise RuntimeError("[ERROR]: must specify start or stop for slice")

                l_bottom = self.getbottom(l_input, type_name)
                valid_pts = []
                for pt in [start, stop]:
                    # print "point :",pt
                    if pt is not None and pt != 0 and pt != l_input.size()[1]:
                        valid_pts.append(pt)

                if len(valid_pts) <= 0:
                    raise RuntimeError("[ERROR]: slice is invalid.")
                elif len(valid_pts) == 1:
                    # if stop != 0 and stop is not None and stop != l_input.size()[1]:
                    if stop in valid_pts:
                        tops = L.Slice(l_bottom, name=name, slice_dim=1, slice_point=valid_pts, ntop=len(valid_pts) + 1)
                        self.frame_v[0].append(name)
                        self.frame_v[1].append(tops[0])
                        self.frame_match[0].append(tops[0])
                    else:
                        tops = L.Slice(l_bottom, name=name, slice_dim=1, slice_point=valid_pts, ntop=len(valid_pts) + 1)
                        self.frame_v[0].append(name)
                        self.frame_v[1].append(tops[1])
                        self.frame_match[0].append(tops[1])
                elif len(valid_pts) == 2:
                    tops = L.Slice(l_bottom, name=name, slice_dim=1, slice_point=valid_pts, ntop=len(valid_pts) + 1)
                    self.frame_v[0].append(name)
                    self.frame_v[1].append(tops[1])
                    self.frame_match[0].append(tops[1])

                self.frame_match[1].append(arg)
                self.frame_match[2].append(id(arg))

                self.message(frame.f_code.co_name + "(slice)", type_name, 'ok')
                # raw_input()

            elif frame.f_code.co_name == "chunk":
                self.r1 += 1
                name = 'slice' + `self.r1`
                dim = frame.f_locals["dim"]  # 作用域下的参数
                chunks = frame.f_locals["chunks"]  # 作用域下的参数
                split_size = frame.f_locals["split_size"]  # 作用域下的参数
                l_input = frame.f_locals["tensor"]  # 作用域下的参数
                type_name = 'Slice'
                l_bottom = self.getbottom(l_input, type_name)
                l_tops = [0 for x in range(chunks)]
                for iterator in range(chunks):
                    l_tops[iterator] = L.Slice(l_bottom, name=name, slice_param=dict(axis=dim, slice_point=split_size))
                    self.frame_v[0].append(name)
                    self.frame_v[1].append(l_tops[iterator])
                    self.frame_match[0].append(l_tops[iterator])
                    self.frame_match[1].append(arg)
                    self.frame_match[2].append(id(arg))

                self.message(frame.f_code.co_name, type_name, 'ok')
                l_output = frame.f_locals["result"]

            elif frame.f_code.co_name == "max_pool2d":
                self.o_max += 1
                name = "maxpool" + `self.o_max`
                type_name = "MTPooling"
                kernel_size1 = frame.f_locals["kernel_size"]
                pad1 = frame.f_locals["padding"]
                stride1 = frame.f_locals["stride"]
                return_indices = frame.f_locals["return_indices"]
                if self.log_flag == True:
                    print("-------------------converting {}: {}".format(name, frame.f_code.co_name))
                    print("kernel,pad,stride,return_indices", kernel_size1, pad1, stride1, return_indices)
                if stride1 == None:
                    stride1 = kernel_size1
                pool1 = P.Pooling.MAX

                l_input = frame.f_locals["input"]
                l_bottom = self.getbottom(l_input, type_name)

                if return_indices:
                    if l_bottom == self.l_data:
                        l_top, l_top1 = L.MTPooling(name=name, bottom="data",
                                                    pooling_param=dict(pool=pool1, kernel_size=kernel_size1,
                                                                       stride=stride1, pad=pad1), ntop=2)
                    else:
                        l_top, l_top1 = L.MTPooling(l_bottom, name=name,
                                                    pooling_param=dict(pool=pool1, kernel_size=kernel_size1,
                                                                       stride=stride1, pad=pad1), ntop=2)
                    self.frame_match[0].append(l_top)
                    self.frame_match[1].append(arg[0])
                    self.frame_match[2].append(id(arg[0]))
                    self.frame_match[0].append(l_top1)
                    self.frame_match[1].append(arg[1])
                    self.frame_match[2].append(id(arg[1]))
                else:
                    if l_bottom == self.l_data:
                        l_top = L.MTPooling(name=name, bottom="data",
                                            pooling_param=dict(pool=pool1, kernel_size=kernel_size1, stride=stride1,
                                                               pad=pad1))
                    else:
                        l_top = L.MTPooling(l_bottom, name=name,
                                            pooling_param=dict(pool=pool1, kernel_size=kernel_size1, stride=stride1,
                                                               pad=pad1))
                        self.frame_match[0].append(l_top)
                    self.frame_match[1].append(arg)
                    self.frame_match[2].append(id(arg))
                self.frame_v[0].append(name)
                self.frame_v[1].append(l_top)
                self.message(frame.f_code.co_name, type_name, 'ok')
            elif os.path.dirname(frame.f_back.f_code.co_filename) == self.target_dir:
                print colored('[Not support]', 'red'), colored(
                    '{} in {}'.format(frame.f_code.co_name, frame.f_back.f_code.co_filename), 'blue')
                sys.exit(-3)
        return self.traceit

    def make_testable(self,train_model_path):
        with open(train_model_path) as f:
            train_str = f.read()
        train_net = caffe_pb2.NetParameter()  # to expose layer.type for future processing
        text_format.Merge(train_str, train_net)
        return train_net

    def save_caffemodel(self):
        caffe_proto = self.deploy
        caffe_net = caffe.Net(caffe_proto, caffe.TEST)
        if self.log_flag == True:
            print 'load to test:', caffe_proto
            print 'load proto ok. ', caffe_net
            print("===1. load prototxt ok================================")
        if self.log_flag == True:
            print(type(caffe_net.params))
            print("====2. params coping================================")
        model_in = self.make_testable(caffe_proto)
        for layer in model_in.layer:
            if layer.type == 'Convolution' or layer.type == 'Deconvolution':
                if self.log_flag == True:
                    print(' ---------------------------------------------copying ' + layer.name)
                    # print(py_params[layer.name][0])
                    # print(dir(py_params[layer.name][1]))
                    print(type(self.py_params[layer.name][0].data.double()))
                    print(len(self.py_params[layer.name]))
                    print layer.name
                    print "caffe shape:", caffe_net.params[layer.name][0].data.shape
                    print "pytorch shape:", self.py_params[layer.name][0].data.double().numpy().shape
                if (len(self.py_params[layer.name]) > 1):
                    caffe_net.params[layer.name][0].data[...] = self.py_params[layer.name][0].data.double().numpy()
                    caffe_net.params[layer.name][1].data[...] = self.py_params[layer.name][1].data.double().numpy()
                elif (len(self.py_params[layer.name]) == 1):
                    caffe_net.params[layer.name][0].data[...] = self.py_params[layer.name][0].data.double().numpy()
            elif layer.type == 'BatchNorm':
                if self.log_flag == True:
                    print (' ---------------------------------------------copying ' + layer.name)
                caffe_net.params[layer.name][0].data[...] = self.py_params[layer.name][0].double().numpy()
                caffe_net.params[layer.name][1].data[...] = self.py_params[layer.name][1].double().numpy()
                caffe_net.params[layer.name][2].data[...] = 1.0
            elif layer.type == 'Scale':
                if self.log_flag == True:
                    print "scale shape:", caffe_net.params[layer.name][0].data.shape
                    print (' ---------------------------------------------copying ' + layer.name)
                if (len(self.py_params[layer.name]) > 1):
                    caffe_net.params[layer.name][0].data[...] = self.py_params[layer.name][0].data.double().numpy()
                    caffe_net.params[layer.name][1].data[...] = self.py_params[layer.name][1].data.double().numpy()
                else:
                    caffe_net.params[layer.name][0].data[...] = self.py_params[layer.name][0]
            elif layer.type == 'PReLU':  # prelu
                if self.log_flag == True:
                    print (' ---------------------------------------------copying ' + layer.name)
                caffe_net.params[layer.name][0].data[...] = self.py_params[layer.name][0].data.double().numpy()
            elif layer.type == 'InnerProduct':
                if self.log_flag == True:
                    print (' ---------------------------------------------copying ' + layer.name)
                if (len(caffe_net.params[layer.name]) > 1):
                    caffe_net.params[layer.name][0].data[...] = self.py_params[layer.name][0].data.double().numpy()
                    caffe_net.params[layer.name][1].data[...] = self.py_params[layer.name][1].data.double().numpy()
                else:
                    caffe_net.params[layer.name][0].data[...] = self.py_params[layer.name][0].data.double().numpy()

        caffe_net.save(self.caffe_file_name)

    ####=====================================================================================
    def main(self,model, images):

        # model.eval()
        print("=================convert ========================")
        sys.settrace(self.traceit)
        output = model(images)
        # getbottom(output,"last layer")
        sys.settrace(None)
        print("=================convert complete=================")
        if self.log_flag == True:
            print("==input image==")
            print(images)
            print("==output Variable==")
            print(output)

        if self.log_flag == True:
            print("========= =====2 frame_v======================")
            print(self.frame_v)
            print("===============3 frame_match===================")
            print self.frame_match
            print(type(self.frame_match))

        with open(self.deploy, 'w') as f:
            f.write('name:"' + self.file_name + '"\n')
            f.write('layer {\n')
            f.write('name:"data"\n')
            f.write('type:"Input"\n')
            f.write('top:"data"\n')
            f.write('input_param { shape : {')
            f.write('dim: %d ' % images.size(0))  # num  可自行定义
            f.write('dim: %d ' % images.size(1))  # 通道数 表示RGB三个通道
            f.write('dim: %d ' % images.size(2))  # 图像的长和宽，通过*_train_text.prototxt 文件中数据输入层的crop_size 获取
            f.write('dim: %d ' % images.size(3))
            f.write('} }\n')
            f.write('}\n')
            f.write(str(to_proto(self.frame_v[1][-1])))

        if self.log_flag == True:
            print("=================save prototxt ok.=================")
            print("=================do savecaffemodel=================")
        self.save_caffemodel()
        if self.log_flag == True:
            print("================ save caffemodel ok================")

    def test_convert(self,images):
        if log_flag == True:
            print("=================test ========================")
            print "====1、test in pytorch:====="
        import utils
        utils.log_flag = log_flag
        for v in output:
            output_pytorch = v.data.numpy()
            v = output_pytorch.flatten()
            i_min = v.argmin()
            i_max = v.argmax()
            if log_flag == True:
                print '---------------------'
                print ("the pre ten number: ", v[:10])
                print("max index :", i_max)
                print("max number :", v[i_max])
                print("min index :", i_min)
                print("min number :", v[i_min])
                print("the pytorch test  end")
                print '---------------------'
        i_max_pytorch = i_max
        i_min_pytorch = i_min

        if log_flag == True or 1:
            print "====2、test in caffe:====="
        input_image = images.data.numpy()
        if log_flag == True:
            print input_image
            print dir(input_image)
        output_caffe, i_max_caffe, i_min_caffe = utils.test_caffe(input_image)
        print "caffe out shape:", output_caffe.shape
        sub_res = output_pytorch - output_caffe
        sub_max = sub_res.max()
        print "max diff :", sub_max
        if sub_max < 1e-5 and i_max_caffe == i_max_pytorch and i_min_caffe == i_min_pytorch:
            print colored('convert success.', 'green')
            print "prototxt file : ", deploy
            print "caffemodel file: ", caffe_file_name
        else:
            print colored('convert failed !', 'red')

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Transform from pth to caffemodel')
        parser.add_argument('--sys_root', type=str, default="xxx",
                            help='set the pth name examples/En_De_Code/En_De_Code.py')
        parser.add_argument('--model_name', type=str, default="xxx.pth",
                            help='set the model name in examples/En_De_Code')
        parser.add_argument('--structure_model_name', type=str, default="AE_ud3_halfx2",
                            help='set the model name in examples/En_De_Code')

        return parser.parse_args()

    ####=====================================================================================

    def Converter(self,model, images, file_path, file_name):
        self.l_bottom, \
        self.l_top, \
        self.l_top1, \
        self.l_data, \
        self.output, \
        self.target_dir
        self.i1, \
        self.i2, \
        self.i3, \
        self.i4, \
        self.i5, \
        self.j, \
        self.k, \
        self.k1, \
        self.k2, \
        self.k3, \
        self.l, \
        self.p, \
        self.p1, \
        self.q, \
        self.o, \
        self.o2, \
        self.o3, \
        self.o4, \
        self.o5, \
        self.o_s, \
        self.o_max, \
        self.r, \
        self.r1 = (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        l_bottom = 0
        l_top = 0

        self.frame_v = [[], []]
        self.frame_match = [[], [], []]
        self.py_params = {}
        self.target_dir = ''
        self.log_flag = False
        self.file_name = file_name
        self.deploy = file_path + '/' + file_name + '.prototxt'  # prototxt文件的保存路径
        self.caffe_file_name = file_path + '/' + file_name + '.caffemodel'  # caffemodel文件的保存路径

        if self.log_flag == True:
            print ("file_path :", file_path)  # 网络类文件的文件路径
            print ("file_name :", file_name)  # 网络类的文件名
            print ("deploy prototxt file:", self.deploy)
            print ("caffe_file_name caffemodel file:", self.caffe_file_name)
        self.target_dir = os.path.dirname(file_path + '/' + file_name + '.py')
        sys.path.append(os.path.dirname(file_path + '/' + file_name + '.py'))

        model.cpu()

        name = "data"
        l_top = L.Input(name=name, input_param=dict(
            shape=dict(dim=[images.size(0), images.size(1), images.size(2), images.size(3)])))
        l_data = l_top
        self.frame_v[0].append(name)
        self.frame_v[1].append(l_top)
        self.frame_match[0].append(l_top)
        self.frame_match[1].append(images)
        self.frame_match[2].append(id(images))
        print type(images)
        self.main(model, images)





if __name__ == '__main__':
    Converter(model, images, file_path, file_name)
