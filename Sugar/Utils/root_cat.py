

import os
import shutil
import random
import numpy as np
import cv2
A_path = '/root/group-dia/wssy/ReLink/Disney_v3/dataset'

#B_path = '/root/group-dia/DIA/styles/Disney/all/source'
B_path = '/root/group-dia/wssy/ReLink/Disney_source/src'
#B_path = '/root/group-dia/rsh/DIA/gen/Naren/iteration_v2/output_with_super_mask_d3110/A/train'
A_list = os.listdir(A_path)
B_list = os.listdir(B_path)
save_path = '/root/cat_dir/A'



file_list = []



def func(mask):
    b = mask[:, :, 0]
    g = mask[:, :, 1]
    r = mask[:, :, 2]
    blue = np.ones(g.shape)*255
    face = g==202
    hair = g == 158
    cloth= np.bitwise_or(g==206,b==255)
    bg = g==0

    blue[bg] = 0
    green = blue.copy()
    red = blue.copy()
    red[:]=0
    red[hair]=255
    red[face] = 255
    blue[face] =0
    green[hair] = 0
    green[cloth] =0


    mask_ = cv2.merge([blue,green,red])
    return mask_.astype(np.uint8)

for i,name in enumerate(B_list):
    local_path = os.path.join(B_path,name)

    mask_p = local_path.replace('src','mask')
    print 'load:', i, mask_p
    tar_p = local_path.replace('src', 'tar')

    src = cv2.imread(local_path)
    mask = cv2.imread(mask_p)
    tar = cv2.imread(tar_p)

    item = [
        src, mask, tar
    ]
    h,w = src.shape[:2]
    scale = 1280./max(h,w)
    w_ = int(w * scale + 0.5)
    h_ = int(h * scale + 0.5)
    item = [cv2.resize(x,(w_,h_)) for x in item]

    file_list.append(item)

            #save_p = os.path.join(save_path,str(count)+'.png')
            #shutil.copyfile(src_p, save_p)

