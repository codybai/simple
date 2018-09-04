import torch
import os
import numpy as np
from torch.autograd import Variable
class LogEditor():
    def __init__(self,root):
        self.ReservedWord = 'Loss,Main,Image'
        from Loop.ShowImage import Cv2Show
        self.LogOutput = Cv2Show()
        self.str_l = ['after','before' ]
        self.dict = None

        self.save_image = os.path.join(root,'Image')
        self.save_test = os.path.join(root, 'Test')
        self.save_cat = os.path.join(self.save_test, 'Cat')
        self.save_bf = os.path.join(self.save_test, 'BeforeAfter')
        self.save_diy = os.path.join(self.save_test, 'DIY')
        self.save_caffe = os.path.join(self.save_test, 'Caffe')
        self.makedirs(root)

    def makedirs(self,root):
        path_list = [
            root,
            self.save_image,
            self.save_test,
            self.save_cat,
            self.save_bf,
            self.save_diy,
            self.save_caffe
        ]
        for p in path_list:
            if not os.path.exists(p):
                os.makedirs(p)

    def ConcatImageMat(self,im_list,cat_dim):
        group_list = []
        for group in im_list:
            image_1 = torch.cat(group, cat_dim)
            group_list.append(image_1)
        image = torch.cat(group_list, 5 - cat_dim)

        return self._tensor_2_image(image)

    def _tensor_2_image(self,image):
        if isinstance(image,Variable):
            flag = '.jpg'
            image = image.data
        else:
            flag = '.png'

        cpu_var = image.cpu()
        tensor = cpu_var[0]
        arr = tensor.numpy()
        im = arr.transpose(1, 2, 0)
        im = np.clip((im + 1) * 127.5, 0, 255).astype(np.uint8)
        im = im[:,:,::-1]
        return im,flag

    def Log(self,e,ii,num,loss,time,max_len = 100):
        loss_str = ''
        max_v = max(loss)
        if self.dict is None:
            self.dict ={}
            for i in range(len(loss)):
                self.dict[str(i)] = []
                self.dict['v'+str(i)] = []
        total_str = ''
        mean_str = ''
        for i,l in enumerate(loss):
            str_i = str(i)
            if max_v ==l:
                loss_str += '[%.1f]' % (l*10)
                self.dict[str_i].append(1)
            else:
                loss_str+=' %.1f ' % (l*10)
                self.dict[str_i].append(0)
            self.dict['v' + str_i].append(l)
            start = min(len(self.dict[str_i]),max_len)
            total_str+=' '+str(sum(self.dict[str_i][-start:]))
            avg = sum(self.dict['v' + str_i][-start:])/float(len(self.dict['v' + str_i][-start:]))
            mean_str += '%.2f '%(100*avg)
        self.LogOutput.Show(self.dict['v0'])
        return '[%d-%d/%d]%.5f:%s%s|%s' %(e,ii,
                                          num,
                                          time,
                                          loss_str,
                                          total_str,
                                          mean_str)

    def OutImage(self,main_image,name):
        if isinstance(name, int):
            name = str(name)

        if isinstance(main_image, list):
            im_list = []
            for ii, img in enumerate(main_image):
                main_img,flag = self._tensor_2_image(img)
                sample_path = os.path.join(
                    self.save_bf, name + '_' + self.str_l[ii] + flag)

                self.LogOutput.Save(sample_path, main_img)
                im_list.append(main_img)
            cat_path = os.path.join(self.save_cat, name + flag)
            self.LogOutput.Save(cat_path, np.concatenate(im_list, 1))
        else:
            main_img,flag = self._tensor_2_image(main_image)
            sample_path = os.path.join(self.save_test, name + '_S'+flag)
            self.LogOutput.Save(sample_path, main_img)

    def PrintLog(self,Log_dict,epoch,iter,len_dl,time_):
        log = Log_dict['Loss']
        log_str = self.Log(epoch,
                           iter,
                           len_dl,
                           log,
                           time_ )

        self.LogOutput.PrintLog(log_str)

    def Finder(self,Log_dict,name,epoch,iter,len_dl,time_,cat_dim):
        if Log_dict is None:
            return
        log = Log_dict['Loss']
        im_list = Log_dict['Image']
        im,flag = self.ConcatImageMat(im_list,cat_dim)

        save_path = os.path.join(self.save_test,
                                 str(name) + flag)
        self.LogOutput.Save(save_path, im)

        if 'Main' in Log_dict:
            Main_Image = Log_dict['Main']
            self.OutImage(Main_Image,name)

        log_str = self.Log(epoch,
                           iter,
                           len_dl,
                           log,
                           time_ * 1000,
                           1e8)

        self.LogOutput.PrintLog(log_str)
        if isinstance(name, int):
            name = str(name)
        for key in Log_dict:
            if key not in self.ReservedWord:
                img,flag = self._tensor_2_image(Log_dict[key])
                local_path = os.path.join(self.save_diy,name+'_'+key+flag)
                self.LogOutput.Save(local_path, img)

    def Shower(self,Log_dict,cat_dim,e,count,len_dl):
        image_list = Log_dict['Image']
        if len(image_list)==0:
            return

        group_list = [
            torch.cat(group, cat_dim)[0:1]for group in image_list
        ]
        image = torch.cat(group_list, 5 - cat_dim)

        im,flag = self._tensor_2_image(image)

        save_path = os.path.join(
            self.save_image,
            str(int(count * 100 / len_dl)) + '_' + str(e) + flag)

        self.LogOutput.Save(save_path, im)

        if 'Main' in Log_dict:
            if isinstance(Log_dict['Main'], list):
                main_img,flag = self._tensor_2_image(Log_dict['Main'][0])
            else:
                main_img,flag = self._tensor_2_image(Log_dict['Main'])
            return self.LogOutput.Main(im,main_img)
        else:
            return self.LogOutput.Grid(im)

