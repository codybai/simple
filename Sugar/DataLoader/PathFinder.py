import os
import cv2
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
])

def flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, i in enumerate(input_list):

            if type(i) == list:
                input_list = i + input_list[index + 1:]
                break
            else:
                output_list.append(i)
                input_list.pop(index)
                break

    return output_list

def Find_key(dir,key_name):
    file_list = os.listdir(dir)
    path_list = []
    if key_name[0]=='/':
        for name in file_list:
            path = os.path.join(dir, name)
            if key_name+'/' in path+'/' :
                return path

            if os.path.isdir(path):
                path_list.append(Find_key(path,key_name))
    else:

        for name in file_list:
            path = os.path.join(dir, name)
            if key_name in name :
                if '.' in name:
                    return  dir
                return path
            if os.path.isdir(path):
                path_list.append(Find_key(path,key_name))

    return flatten(path_list)

def GetFile(root,name_list,key_name,out_list):
    if os.path.isdir(root):
        file_list = os.listdir(root)
        for name in file_list:
            path = os.path.join(root, name)
            GetFile(path, name_list,key_name,out_list)
    else:
        ll = []
        for name in name_list:
            if key_name in root:
                local_path = root.replace(key_name, name)

                if os.path.exists(local_path):
                    ll.append(local_path)
                else:
                    return
        if len(ll)>0:
            out_list.append(ll)

def make_dataset(dir,key_name,name_list):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    path_list = Find_key(dir,key_name)
    if isinstance(path_list,list)==False:
        path_list = [path_list]

    for root in path_list:
        print 'Loading the dir...',root
        GetFile(root,name_list,key_name,images)
    return images

class SingleDataset():
    def initialize(self, dataroot,key_name,name_list,pre_p=None,group_p=None,isTest=False):

        self.root = dataroot
        self.name_list = name_list
        self.key_name = key_name
        self.dir_A = os.path.join(dataroot)
        self.pre_p = pre_p
        self.group_p = group_p
        self.test = isTest
        if name_list is not None:
            self.A_paths = make_dataset(self.dir_A,key_name,name_list)
            self.A_paths = sorted(self.A_paths)
        else:
            self.A_paths = []

            for filename in os.listdir(self.root):
                x = os.path.join(self.root, filename)
                if self.key_name  in x:
                    self.A_paths.append([x])
        #self.transform = get_transform(opt)

    def __getitem__(self, index):

        load_list = [cv2.imread(path) for path in self.A_paths[index]]
        f_name = self.A_paths[index][0].split('/')[-1]
        str_name = ''
        for s in f_name.split('.')[:-1]:
            str_name+=s

        for i,x in enumerate(load_list):
            if hasattr(x,'shape')==False:
                print self.A_paths[index][i]
                load_list = [cv2.imread(path) for path in self.A_paths[0]]
                break

        if self.group_p is not None:
            load_list = self.group_p(load_list)
            if load_list is None:
                return None
        output_list = []
        for img in load_list:
            if  len(img.shape)==3 :
                if self.pre_p is not None:
                    img = self.pre_p(img)
                if img.shape[2] ==3:
                    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                else:
                    rgb_img = img
                chw = rgb_img.transpose(2, 0, 1).astype(np.float32)
                chw = chw / 127.5 - 1
            else:
                chw = img
            output_list.append(chw)
        if self.test:
            return output_list,str_name
        return output_list

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'A_TorchImageDataset'

