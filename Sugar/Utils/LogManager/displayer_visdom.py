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
class displayer_visdom():
    def __init__(self,port):
        self.vis = visdom.Visdom(port=port)

    def display_line(self,X_list,Y_list,name,legend_list,win_id):
        line_num = len(Y_list)
        self.vis.line(
            X=np.stack([np.array(X_list)] * line_num, 1),
            Y=np.stack(np.array(Y_list), 1),
            opts={
                'title': name + ' loss over time',
                'legend': legend_list,
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=win_id)

    def display_image(self,img,label,win_id):
        self.vis.image(img, opts=dict(title=label),
                       win=win_id)

    def display_str(self,message,win_id):
        self.vis.text(message,win_id)
