import cv2
import numpy as np
import thread

def get(arr):
    len_n = len(arr)
    if len_n>=200:
        len_n-=20
        A_part = arr[:-20]
        B_part = arr[-20:]

        step = len_n / 180.

        list = []
        for i in range(180):
            index = step * i
            ii = min(len_n, int(index))
            list.append(A_part[ii])
        list+=B_part
    else:
        list = [0]*(200-len_n)+arr

    return np.array(list)


def update(frame):
    global dis
    global dis_num
    global line
    if len(dis)!=dis_num:
        dis_num = len(dis)
        line.set_ydata(get(dis))
    return line

class Cv2Show():
    def __init__(self):
        #thread.start_new_thread(drawline,())
        pass

    def Show(self,dis2_):
        global dis
        dis = dis2_

    def Save(self,save_path,im):
        cv2.imwrite(save_path, im)
    def PrintLog(self,log_str):
        print log_str
    def Grid(self,im):
        h, w = im.shape[:2]
        scale_ = min(1720. / h, 1050. / w)
        h_ = int(h * scale_)
        w_ = int(w * scale_)
        im = cv2.resize(im, (w_, h_))
        self.Display_Image(im)
        return im

    def Display_Image(self, img,epoch=0):
        cv2.imshow('',img)
        cv2.waitKey(1)
        pass
        #self.imw.setImage(img,xvals=np.linspace(1., 3., img.shape[0]))

    def Main(self,im,main_img):
        h, w = im.shape[:2]
        ph, pw = main_img.shape[:2]
        rw = max(10, 1080 - pw)
        rh = rw * h / w
        all_h = max(ph, rh)
        multi = cv2.resize(im, (rw, rh))
        fill_map = np.zeros((all_h, 1080, 3), dtype=np.uint8)
        fill_map[:rh, :rw] = multi
        fill_map[:ph, rw:1080] = main_img[:, :1080 - rw]
        del main_img
        im = fill_map

        self.Display_Image(im)

        return im

