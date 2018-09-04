import os
class OptionBase():
    def __init__(self,params=None):
        self.params = {
            'Root':           '/root/group-dia/wssy/ReLink',
            'key_name':       '',
            'input_dir':      'Disney/src/,'
                              'Disney/tar/,'
                              'Milk/tar/,'
                              'Ink/tar/',
            'TestRoot':        '/root/group-dia/wssy/ReLink/test_part',
            'Testkey_name':    '',
            'Testinput_dir':   '/A,'
                               '/AP',
            'Classification':  False,
            'Test':            False,
            'ToCaffe':         False,
            'checkpoints':     '/root/checkpoints',
            'batch_size':       1,
            'continue_train':   False,
            'loops':            0,
            'epoch':            100,
            'epoch_model':      -1,
            'lr':               1e-4,
            'vggLayerNames':'conv3_1,conv3_2,'
                            'conv4_1,conv4_2,'
                            'conv4_3,conv4_4,'
                            'conv5_1,conv5_2',
            'BufferMode':   2,
            'show_scale':   2,
            'nThreads'   :  8,
            'gpu_id_list': [0],
            'max_batch':    1e8,
            'show_iter':    50,
            'shuffle':      False,
            'multi_rank': 0,
            'multi_size': 0,
            'master_addr': ' 10.244.26.12',
            'loss':         '1*L1(O_1,T_1)',
            'hardcase'  :   False,
            #if you want to open it,you need set the param by 'This is a bad solution to show the results.'
            #
            #'open_visdom': 'This is a bad solution to show the results.',
            'open_visdom': 'No',
        }
        if params is not None:
            for name in params:
                if name in self.params:
                    self.params[name] = params[name]
            cp = self.params['checkpoints']
            if 'UsrName' in cp:
                name = self.find_name()
                if name !=False:
                    self.params['checkpoints'] = cp.replace('UsrName',name)

        self.add_members()
    def find_name(self):
        member_list = [
            'hwd',
            'cyw',
            'rsh',
            'wxj',
            'wx',
            'wym',
            'wssy',
            'yzl'
        ]
        for n in member_list:
            path_ = os.path.join('/root',n)
            if os.path.exists(path_):
                return n
        return False
    def add_members(self):
        for name in self.params:
            setattr(self,name,self.params[name])