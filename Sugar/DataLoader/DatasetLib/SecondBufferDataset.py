import os
import pickle
import cv2
import numpy as np
from ..DatasetLib.BufferDataset import BufferDataset

class SecondBufferDataset(BufferDataset):
    def __init__(self,classification):
        super(SecondBufferDataset, self).__init__(classification)
        self.OverSign = '_Over'

    def save_data(self,output_list,path_dir):
        output_num = len(output_list)
        for i in range(output_num):
            out = output_list[i]
            index_name ='_' + str(i)
            if i == output_num-1:
                index_name+=self.OverSign
            if isinstance(out,np.ndarray) and out.dtype == 'uint8':
                path_ = os.path.join(path_dir,
                                     index_name  + '.png')
                cv2.imwrite(path_,out)
            else:
                path_ = os.path.join(path_dir,
                                     index_name + '.atp')
                byte_data = pickle.dumps(out)
                f = open(path_,'w')
                f.write(byte_data)
                f.close()

    def isExists(self,path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
            return False

        file_list = os.listdir(path_dir)
        GetOver = False
        for path_file in file_list:
            if self.OverSign in path_file:
                GetOver = True
        return GetOver

    def __getitem__(self, index):
        path_dir = os.path.join(self.buffer_path, str(index))
        if self.isExists(path_dir):
            output_list = []
            file_list = os.listdir(path_dir)
            over = True
            for path_file in file_list:
                path_ = os.path.join(path_dir, path_file)
                if '.png' in path_:
                    image = cv2.imread(path_)
                    if image is None:
                        over = False
                        break
                    output_list.append(image)
                else:
                    f = open(path_, 'r')
                    byte_data = f.read()
                    f.close()
                    output_list.append(pickle.loads(byte_data))
            if over:
                return self.Normalization(output_list)

        data_list = [self.get_note(cv2.imread(path),path) for path in self.A_paths[index]]
        for d in data_list:
            if d is None:
                print self.A_paths[index]
                data_list = [self.get_note(cv2.imread(path),path) for path in self.A_paths[0]]
                break
        output_list = self.process_data(data_list)
        self.save_data(output_list,path_dir)

        return self.Normalization(output_list)


    def name(self):
        return 'A_TorchImageDatasetSecondBuffer'
