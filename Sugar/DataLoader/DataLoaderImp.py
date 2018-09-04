from DataLoaderBase import DataLoaderBase
import numpy as np

def cuda_(x):
    return x.cuda()

def float_(x):
    return x.astype(np.float32)


def uint8_one_(x):
    return x/127.5-1

class DataLoaderImp(DataLoaderBase):
    def __init__(self,opt,pre_p=None,group_p=None):
        super(DataLoaderImp,self).__init__(opt.BufferMode,opt.Classification)
        if opt.Test:
            root = opt.TestRoot
            k_name = opt.Testkey_name
            input_dir =opt.Testinput_dir
            batch_size = 1
            shuffle = True
        else:
            root = opt.Root
            k_name = opt.key_name
            input_dir = opt.input_dir
            batch_size = opt.batch_size
            shuffle = opt.shuffle

        if input_dir is not None:
            name_list = input_dir.split(',')
        else:
            name_list = None
        if k_name is None or len(k_name)==0:
            if name_list is not None:
                key_name = name_list[0]
            else:
                key_name = ''
        else:
            key_name = k_name


        self.SetRoot_ex(root,
                        key_name,
                        name_list,
                        batch_size,
                        shuffle,
                        opt.nThreads,
                        opt.max_batch,
                        pre_p,group_p,opt.Test)
        self.pre_list = [float_,uint8_one_]
