import torch
import torch.utils.data
from random import Random
from DataLoader.DatasetLib.FirstBufferDataset import FirstBufferDataset
from DataLoader.DatasetLib.SecondBufferDataset import SecondBufferDataset

from DataLoader.DatasetLib.SingleDataset import SingleDataset
from DataLoader.DatasetLib.SingleDataset import SingleDatasetWithName

class CustomDatasetDataLoader():
    def __init__(self,buffermode,classification):
        if buffermode<=0:
            self.dataset = SingleDataset(classification)
        elif buffermode == 1:
            self.dataset = FirstBufferDataset(classification)
        elif buffermode == 2:
            self.dataset = SecondBufferDataset(classification)
        self.batchSize = 1

    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self,
                   optroot,
                   key_name,
                   name_list,
                   batchSize,
                   serial_batches,
                   nThreads,
                   max_dataset_size,
                   pre_p,group_p):
        self.max_dataset_size = max_dataset_size

        print("dataset [%s] was created" % (self.dataset.name()))
        self.dataset.initialize(optroot,
                                key_name,
                                name_list,
                                pre_p,
                                group_p)
        self.batchSize = batchSize

        torch.initial_seed()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batchSize,
            shuffle=not serial_batches,
            num_workers=int(nThreads)
        )

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset)//self.batchSize, self.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.max_dataset_size :
                break
            yield data

class DataLoaderBase(object):
    def __init__(self,buffermode,classification):
        self.files_list = []
        self.batch_list = []
        self.batch_size = 1
        self.count = 0
        self.num = 0
        self.batch_num = 0
        self.data_iter = CustomDatasetDataLoader(buffermode,classification)
        self.classification = classification
    def SetRoot_ex(self,
                   root,
                   key_name,
                   name_list,
                   batch_size,
                   shuffle,
                   nThreads,
                   max_batch,
                   pre_p,
                   group_p,isTest):
        if isTest:
            self.data_iter.dataset = SingleDatasetWithName(self.classification)
        self.data_iter.initialize(root,
                                  key_name,
                                  name_list,
                                  batch_size,
                                  shuffle,
                                  nThreads,
                                  max_batch,
                                  pre_p,
                                  group_p)
        self.data_list = self.data_iter.load_data()


if __name__  =='__main__':
    db =  DataLoaderBase()

    opt = {
        'Root': '/root/group-dia/wssy/ReLink',
        'input_dir': 'Disney/src/,'
                     'Disney/tar/,'
                     'Milk/tar/,'
                     'Ink/tar/',
        'batch_size': 2,
        'continue_train': True,
        'vggLayerNames': 'conv3_1,conv3_2,'
                         'conv4_1,conv4_2,'
                         'conv4_3,conv4_4,'
                         'conv5_1,conv5_2',

        'loss': '1*VGG(O_1,T_1)-'
                '1000*L1(O_1,T_1)-'
                '0.1*tv(O_1)-'
                '1000*L2(O_1,T_1)'
    }
    name_list = opt['input_dir'].split(',')
    dir = name_list[0]
