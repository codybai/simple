

import os
import shutil
import random
import numpy as np
import cv2
A_path = '/root/group-dia/wssy/ReLink/Disney_v3/dataset'

#B_path = '/root/group-dia/DIA/styles/Disney/all/source'
B_path = '/root/group-dia/wssy/ReLink/Ink_v4/sfy/girl/v3_crop_inv_warp/A/d_prettygirl'
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

'''
for i,name in enumerate(B_list):
    local_path = os.path.join(B_path,name)

    mask_p = local_path.replace('source','super_mask')
    print 'load:', i, mask_p
    tar_p = local_path.replace('source', 'style_v1/face')

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
    file_list.append([cv2.resize(x,(w_,h_)) for x in item])

            #save_p = os.path.join(save_path,str(count)+'.png')
            #shutil.copyfile(src_p, save_p)
   '''
for i,name in enumerate(B_list):
    local_path = os.path.join(B_path,name)

    mask_p = local_path.replace('/A/','/gt_super_mask/')
    print 'load:', i, mask_p
    tar_p = local_path.replace('/A/', '/AP/')

    src = local_path
    mask = mask_p
    tar = tar_p

    item = [
        src, mask, tar
    ]

    file_list.append(item)
    if i >200:
        break
            #save_p = os.path.join(save_path,str(count)+'.png')
            #shutil.copyfile(src_p, save_p)


random.shuffle(file_list)

select_num = min(100,len(file_list))

file_list = file_list[:select_num]

count = 0
for name in A_list:
    local_path = os.path.join(A_path,name)
    A_file = os.path.join(local_path,'A')
    for root,_,files in os.walk(A_file):
        for f in files:
            src_p =  os.path.join(root,f)
            mask = src_p.replace('/A/','/gt_super_mask/')
            tar = src_p.replace('/A/', '/AP/')
            index = int(np.random.random()*select_num)

            src_2,mask_2,tar_2 = file_list[index]

            save_p = os.path.join(save_path,str(count)+'.png')
            save_m = save_p.replace('/A/','/gt_super_mask/')
            save_t = save_p.replace('/A/', '/AP/')
            save_as = save_p.replace('/A/', '/SrcAndy/')
            save_am = save_p.replace('/A/', '/MaskAndy/')
            save_at = save_p.replace('/A/', '/TarAndy/')

            shutil.copyfile(src_p, save_p)
            shutil.copyfile(mask, save_m)
            shutil.copyfile(tar, save_t)
            if isinstance(src_2,str):
                shutil.copyfile(src_2, save_as)
                shutil.copyfile(mask_2, save_am)
                shutil.copyfile(tar_2, save_at)
            else:
                cv2.imwrite(save_as, src_2)
                cv2.imwrite(save_am, mask_2)
                cv2.imwrite(save_at, tar_2)
            print count
            count+=1