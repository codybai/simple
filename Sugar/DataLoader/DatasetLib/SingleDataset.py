import os
import cv2
import numpy as np
import pickle
import time
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

class SingleDataset(object):
    def __init__(self,classification):
        self.Sugar_dir = '/root/Do_you_need_some_Sugar'
        if not os.path.exists(self.Sugar_dir):
            os.makedirs(self.Sugar_dir)
        if classification:
            self.get_note = self.get_image_path
        else:
            self.get_note = self.get_image

    def initialize(self, dataroot,
                   key_name,
                   name_list,
                   pre_p=None,group_p=None):
        start_time = time.time()
        self.root = dataroot
        self.name_list = name_list
        self.key_name = key_name
        self.dir_A = os.path.join(dataroot)
        self.pre_p = pre_p
        self.group_p = group_p
        path_list_name = self.dir_A+key_name
        if name_list is None:
            path_list_name+='None'
        else:
            for n in name_list:
                if n is None:
                    n = 'None'
                path_list_name+=n
        path_list_name = path_list_name.replace('/','|')
        path_file = os.path.join(self.Sugar_dir,path_list_name+'.sgr')
        if os.path.exists(path_file):
            print 'Found the bufferfile for the path list.',
            f = open(path_file, 'r')
            byte_data = f.read()
            f.close()
            self.A_paths = pickle.loads(byte_data)

        else:
            print 'Not found the bufferfile.',
            if name_list is not None:
                self.A_paths = make_dataset(self.dir_A,key_name,name_list)
                self.A_paths = sorted(self.A_paths)
            else:
                self.A_paths = []
                for filename in os.listdir(self.root):
                    x = os.path.join(self.root, filename)
                    if self.key_name  in x:
                        self.A_paths.append([x])

            byte_data = pickle.dumps(self.A_paths)
            f = open(path_file, 'w')
            f.write(byte_data)
            f.close()

        print 'cost time:',time.time() - start_time, 's'
    def isImage(self,img):
        return hasattr(img,'shape') and len(img.shape) ==3 and img.shape[2] == 3

    def process_data(self,load_list):
        if self.group_p is not None:
            load_list = self.group_p(load_list)
            if load_list is None:
                return None

        if self.pre_p is not None:
            load_list = [self.pre_p(img) for img in load_list]

        output_list = [
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.isImage(img) else img
            for img in load_list
        ]

        return output_list

    def Normalization(self,output_list):
        return [
            out.transpose(2, 0, 1).astype(np.float32)/ 127.5 - 1
            if self.isImage(out) else out
            for out in output_list
        ]

    def get_image(self,image,path):
        return image

    def get_image_path(self,image,path):
        return [image,path]

    def __getitem__(self, index):

        data_list = [
            self.get_note(cv2.imread(path),path)
            for path in self.A_paths[index]
        ]
        for d in data_list:
            if d[0] is None:
                print self.A_paths[index]
                data_list = [self.get_note(cv2.imread(path),path)  for path in self.A_paths[0]]
                break
        output_list = self.process_data(data_list)
        result = self.Normalization(output_list)

        return result

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'A_TorchImageDataset'


class SingleDatasetWithName(SingleDataset):
    def __init__(self,classification):
        super(SingleDatasetWithName, self).__init__(classification)
    def get_filename(self,index):
        f_name = self.A_paths[index][0].split('/')[-1]
        str_name = ''
        for s in f_name.split('.')[:-1]:
            str_name += s
        return str_name
    def __getitem__(self, index):
        str_name = self.get_filename(index)
        data_list = [self.get_note(cv2.imread(path),path) for path in self.A_paths[index]]
        output_list = self.process_data(data_list)
        return self.Normalization(output_list), str_name

    def name(self):
        return 'A_TorchImageDatasetWithName'