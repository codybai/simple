import os
import shutil
import cv2
from ..DatasetLib.BufferDataset import BufferDataset

class FirstBufferDataset(BufferDataset):
    def __init__(self,classification):
        super(FirstBufferDataset, self).__init__(classification)

    def get_data_list(self,index):
        load_list = []
        for i in range(len(self.A_paths[index])):
            name = self.name_list[i].replace('/','')
            path_image = self.A_paths[index][i]
            type_ = path_image.split('.')[-1]
            path_ = os.path.join(self.buffer_path,
                                 str(index) +'_'+name +'.' + type_)
            if not os.path.exists(path_):
                shutil.copy(path_image, path_)
            image = cv2.imread(path_)
            if image is None:
                shutil.copy(path_image, path_)
                image = cv2.imread(path_)
            load_list.append(self.get_note(image,path_image))
        return load_list

    def __getitem__(self, index):
        data_list = self.get_data_list(index)
        output_list = self.process_data(data_list)
        return self.Normalization(output_list)

    def name(self):
        return 'A_TorchImageDatasetFirstBuffer'
