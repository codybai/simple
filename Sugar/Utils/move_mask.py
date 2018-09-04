

import os
import shutil
import random
import numpy as np
import cv2
A_path = '/root/group-dia/wssy/Dataset/multi/'

#B_path = '/root/group-dia/DIA/styles/Disney/all/source'
B_path = '/root/group-dia/wssy/ReLink/Ink_v4/sfy/girl/v3_crop_inv_warp/A/d_prettygirl'
#B_path = '/root/group-dia/rsh/DIA/gen/Naren/iteration_v2/output_with_super_mask_d3110/A/train'
A_list = os.listdir(A_path)
B_list = os.listdir(B_path)
save_path = '/root/group-dia/wssy/Dataset/Multi/Src'

save_path_test = '/root/group-dia/wssy/Dataset/Multi_test/Src'

file_list = []




random.shuffle(file_list)

select_num = min(100,len(file_list))

file_list = file_list[:select_num]

count = 0
for name in A_list:
    local_path = os.path.join(A_path,name)
    for root,_,files in os.walk(local_path):
        for f in files:
            mask_p =  os.path.join(root,f)
            if '_mask.jpg' in mask_p:
                pass
            else:
                continue
            src_p = mask_p.replace('_mask.jpg','.jpg')
            if os.path.exists(src_p):
                pass
            else:
                continue

            f_name = str(count)+'.jpg'
            if '66' in f_name or '88' in f_name:
                temp_path = save_path_test
            else:
                temp_path = save_path
            save_p = os.path.join(temp_path,f_name)
            save_m = save_p.replace('Src','Mask')


            shutil.copyfile(src_p, save_p)
            shutil.copyfile(mask_p, save_m)
            print count
            count+=1