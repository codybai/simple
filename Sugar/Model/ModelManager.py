from .CreateModel import CreateModel
import torch
import os
from ModelWithOptimiz import ModelWithOptimiz
import torch.nn as nn
class ModelManager():
    def __init__(self,opt):
        self.model_list = []
        self.root = opt.checkpoints
        self.gpu_id_list = opt.gpu_id_list
    def Create(self,str_list):
        cm = CreateModel()
        for s in str_list:
            if isinstance(s,str):
                model_from_cm = cm.Create(s)
                mwo = ModelWithOptimiz(model_from_cm)
            elif isinstance(s,nn.Module):
                mwo = s
            else:
                assert False,'I don\'t know how to translate this parameter.'
            self.model_list.append(mwo)

    def cuda(self):
        for m in self.model_list:
            m.cuda(self.gpu_id_list[0])

    def save(self,m,id,count,gpu_ids = [0]):
        save_filename = '%s_net_%s.pth' % (count, id)
        save_path = os.path.join(self.root, save_filename)
        torch.save(m.cpu().state_dict(), save_path)
        if hasattr(m,'cpu_rsh'):
            torch.save(m.cpu_rsh().state_dict(), save_path.replace('.pth','_S.pth'))
        if len(gpu_ids) and torch.cuda.is_available():
            m.cuda(self.gpu_id_list[0])

    def load(self,m,id,count):
        save_filename = '%s_net_%s.pth' % (count, id)

        save_path = os.path.join(self.root, save_filename)

        if os.path.exists(save_path) ==False:
            print save_path +' is not exists.'
            return
        m.load_state_dict(torch.load(save_path))
        if os.path.exists(save_path.replace('.pth','_S.pth')) and hasattr(m,'load_state_dict_rsh'):
            m.load_state_dict_rsh(torch.load(save_path.replace('.pth','_S.pth')))
    def getlist(self):
        for i in range(len(self.model_list)):
            if isinstance(self.model_list[i],ModelWithOptimiz) == False:
                self.model_list[i] = ModelWithOptimiz(self.model_list[i])
        return self.model_list

    def Save(self,count):
        for m_id,m in enumerate(self.model_list):
            self.save(m,m_id,count)

    def Load(self,count):
        for m_id,m in enumerate(self.model_list):
            self.load(m,m_id,count)

