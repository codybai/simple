#!/usr/bin/python
# -*- coding: utf-8 -*-
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 * LogManager类: 用于log的管理
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
import displayer_visdom as displayer_visdom
import PIL.Image as Image
import cv2
class LogManager():
    #need opt:
    #  opt.display_id
    #  opt.displayer
    #  opt.isTrain
    #  opt.no_html
    #  opt.name
    #  opt.display_winsize
    #  opt.txtpath


    def __init__(self, opt):
        # self.opt = opt
        self.display_id = 1
        self.win_size = 256#opt.display_winsize
        self.message = 'This is a string that will be written to .txt\n'
        if not hasattr(opt, 'txtpath'):
            self.path='./log_txt'
        else:
            self.path = opt.txtpath+'/log_txt'
        if self.display_id > 0:
            if not hasattr(opt,'displayer') or opt.displayer == 'visdom':
                if not hasattr(opt,'port'):
                    port = 8008
                else:
                    port = opt.port
                self.displayer = displayer_visdom.displayer_visdom(port)
            else:
                self.displayer =None

    def Display_Line(self,smooth = 1):  # display in web by visdom
        # avgtime=self.get_time()
        win_id = self.display_id

        chart_list = {}
        for k in self.plot_data['legend']:
            for c in self.plot_data[k][1]:
            #    ls = line_smooth.split(':')
            #    line = ls[0]
            #    if len(ls)>1:
            #        n_smooth = int(ls[1])
            #        the_list = self.Smooth(self.plot_data[line],n_smooth)
            #    elif smooth > 1:
            #        the_list = self.Smooth(self.plot_data[line], smooth)
            #    else:
            #        the_list = self.plot_data[line]
                if c not in chart_list:
                    chart_list[c]=[[],[]]
                chart_list[c][0].append(self.plot_data[k][0])
                chart_list[c][1].append(k)

        for chart in chart_list:
            self.displayer.display_line(self.plot_data['X'],
                                        chart_list[chart][0],
                                        chart,chart_list[chart][1],win_id)
            win_id+=1

    # visuals         : it is a dict that contains image_name and image string
    #                   same format as |images| of model.get_current_visuals()
    #                   OrderedDict([('real_A', [real_A,'src']),
    #                                ('fake_B', [fake_B,'out']),
    #                                ('real_B', [real_B,'tar']),
    #                              ])

    def Display_Image(self, img,index = 0 ):
        if self.display_id > 0: # show images in the browser
            win_idx =  self.display_id+index+1000

        self.displayer.display_image(img.transpose(2,0,1),'Image',win_idx)
        win_idx += 1

    # the function will write the message to Console
    # note : same format as |note_list| of self.Record()
    def Print_String(self, epoch, i, note, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)

        for k, v in note.items():
            message += '%s: %.3f ' % (k, v[0])
        self.message += message+'\n'
        print(message)

    # the function will write the message to web
    # note : same format as |note_list| of self.Record()
    def Display_String(self,epoch, i, note, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in note.items():
            message += '%s: %.3f ' % (k, v[0])
            win_id = self.display_id+2000
        self.displayer.display_str(message,win_id)

    # the function will write the message to log.txt
    # the message is the same to the output of Display_String()
    def save_string(self):
        f = open(self.path,'w')
        f.write(self.message)
        f.close()
        print('has written to disk.')

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            #util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
    def Save(self,save_path,im):
        im = Image.fromarray(im)
        im.save(save_path)
    def Main(self,x,m):
        self.Display_Image( x, 0)
        self.Display_Image( m, 1)
    def Grid(self,x):
        self.Display_Image( x, 0)
if __name__ == '__main__':
    __file_name__ = __file__.split('/')[-1]
    task_name = __file_name__.replace('.py','')
    os.environ['TORCH_MODEL_ZOO'] = __file__.replace(__file_name__,
                                            'TORCH_MODEL_ZOO')
    from Option.OptionBase import OptionBase
    opt = OptionBase({
        'Root':         '/root/group-dia/wx/DIA/gen/disney/iteration_v12/girl/v2_v3_combined_sy',
        'key_name':     '/A',
        'input_dir':    '/A,'
                        '/super_mask,'
                        '/AP,'
                        '/gt_sem_mask',

        'TestRoot':     '/root/group-dia/image-site/200testwithmask_milk',
        'Testkey_name': '_input',
        'Testinput_dir':  '_input,'
                        '_hairmask,'
                        '_out.png_ashura0122,'
                        '_hairmask',
        'Test':           False,

        'batch_size':           1,
        'continue_train':    True,
        'vggLayerNames':'conv3_1,'
                        'conv4_1,conv4_2,'
                        'conv4_4',

        'loss':         '0.1*VGG(O_1,T_1)-'
                        '10000*L2(O_1,T_1)',

        'gpu_id_list':  [0],
        'max_batch':    1e8,
        'epoch':        1000,
        'show_iter':    10,
        'checkpoints':  '/root/group-dia/wssy/CheckPointsTemp/'+task_name,
    })

    thvis = LogManager(opt)
    im = np.zeros((3, 3, 3))
    thvis.Display_Image(cv2.resize(im,(5,5)))  # im.transpose(2,0,1))
