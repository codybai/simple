import torch
import copy
class ModelWithOptimiz():
    def __init__(self,model):
        self.model = model
        self.modelS = copy.deepcopy(model)
        self.modelMean = copy.deepcopy(model)
        self.model_list = []
        self.step_count = 0
        self.mean_num = 0.
        self.isUpdate = False
        self.isTest = False
        self.isCaffe = False
        self.Input = None
    def __call__(self, *args, **kwargs):
        if self.isCaffe:
            self.Input = args[0]
        return self.model.__call__(*args, **kwargs)

    def update_modelMean(self):
        self.mean_num+=1
        for param_, param_m in zip(
                self.model.state_dict().values(),
                self.modelMean.state_dict().values()):
            param_m[:] = (param_m*(self.mean_num-1) + param_)/self.mean_num

    def step(self):
        if self.isTest == False:
            self.step_count = (self.step_count+1)%128
            if self.step_count == 0 :
                if len(self.model_list) == 128:
                    self.model_list = self.model_list[1::2]
                self.model_list.append(
                    self.model.state_dict().values()
                )
                self.isUpdate = True
                self.update_modelMean()
            self.optimiz.step()

    def zero_grad(self):
        self.optimiz.zero_grad()

    def Adam(self,lr=0.0004, betas=(0.5, 0.999)):
        self.optimiz = torch.optim.Adam(
            self.model.parameters(),
            lr=lr, betas=betas)
        self.optimiz.zero_grad()

    def cuda(self,id=0):
        self.model.cuda(id)
        self.modelS.cuda(id)
        self.modelMean.cuda(id)

    def update_modelS(self):
        S_state = self.modelS.state_dict()
        if len(self.model_list) > 0:
            for s in S_state.values():
                s[:] = 0
            scale = 1. / len(self.model_list)
            for model_state in self.model_list:
                for param_, param_s in zip(model_state, S_state.values()):
                    param_s[:] = param_s + param_ * scale

            self.modelS.load_state_dict(S_state)

    def ReSampleHarmonic(self, *args, **kwargs):
        if self.isUpdate:
            self.update_modelS()
            self.isUpdate = False
        return self.modelS.__call__(*args, **kwargs)

    def ReSampleHarmonicMean(self, *args, **kwargs):
        return self.modelMean.__call__(*args, **kwargs)

    def cpu(self):
        return self.model.cpu()

    def cpu_rsh(self):
        return self.modelS.cpu()

    def state_dict(self):
        return self.model.state_dict()

    def state_dict_rsh(self):
        self.update_modelS()
        return self.modelS.state_dict()

    def load_state_dict(self,dict):
        self.model.load_state_dict(dict)
        self.modelS.load_state_dict(dict)
        self.modelMean.load_state_dict(dict)
    def load_state_dict_rsh(self,dict):
        self.modelS.load_state_dict(dict)
        self.modelMean.load_state_dict(dict)