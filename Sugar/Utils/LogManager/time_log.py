#!/usr/bin/python
# -*- coding: utf-8 -*-
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 * time_log类: 用于iter_time的管理
 *
 * Copyright (c) 2017年 MTLAB. All rights reserved.
 *
 * @version: 1.0
 *
 * @author: 吴善思源(wssy@meitu.com)
 *
 * @date: 2017-05-18
 *
 * @note:
 *
 * @usage：
 *
 * @change:

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


import numpy as np
import os
import ntpath
import time

import visdom
import math
import time_log as time_log
class time_log():
    def __init__(self,len = 100):
        self.times=[]
        self.num=0
        self.len=len
        self.sum=0
        self.time_list = []
        self.start_tag = False

    def insert(self,val):
        self.times.append(val)
        self.num+=1
        self.sum+=val
        if self.num>self.len:
            self.pop()
        return self.sum/self.num

    def pop(self):
        self.started_or_not()
        self.num-=1
        self.sum-=self.times[0]
        del self.times[0]

    def start(self):
        self.run_time = time.time()
        self.pre_time = self.run_time
        self.start_tag = True

    def get_time(self):
        self.started_or_not()
        self.pre_time = self.run_time
        self.run_time = time.time()
        time_one = self.run_time - self.pre_time
        return self.insert(time_one)

    def record_time(self):

        t = self.get_time()
        self.time_list.append(t)

        return t

    def get_log_list(self):
        return self.time_list

    def started_or_not(self):
        assert self.start_tag == True, 'please call .start() before using'
