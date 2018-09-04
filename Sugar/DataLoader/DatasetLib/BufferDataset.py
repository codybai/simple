import os
from ..DatasetLib.SingleDataset import SingleDataset

class BufferDataset(SingleDataset):
    def __init__(self,classification):
        super(BufferDataset, self).__init__(classification)
        self.buffer_path = os.path.join(self.Sugar_dir,
                        'It_is_A_Torch_Buffer_Space')
        if not os.path.exists(self.buffer_path):
            os.makedirs(self.buffer_path)
    def name(self):
        return 'A_TorchImageDatasetBuffer'