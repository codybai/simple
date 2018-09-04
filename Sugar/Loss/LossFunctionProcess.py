from torch.autograd import Variable
import torch
class LossFunctionProcess():
    def __init__(self):
        self.weight_list =[]
        self.loss_list = []
        self.input_list = []
        self.item_num = 0
        self.index_list = []
        self.current_lossv = 0
    def add_loss(self,weight,loss,p_list):
        self.weight_list.append(weight)
        self.loss_list.append(loss)
        output_list = []
        for p in p_list:
            type_index = p.split('_')
            index = int(type_index[1])
            if type_index[0] == 'O':
                output_list.append([0, index - 1])
            elif type_index[0] == 'T':
                output_list.append([1, index - 1])
            elif type_index[0] == 'N':
                output_list.append([2, index - 1])
            else:
                assert 'type must be O T or N'

        self.input_list.append(output_list)
        self.item_num = len(self.weight_list)
    def Record(self,index):
        self.index_list.append([index,self.current_lossv])
        self.current_lossv = 0
    def Rank(self):
        self.index_list.sort(lambda x,y : cmp(x[1], y[1]))
        output = [x[0] for  x in self.index_list]
        output.reverse()
        self.index_list = []
        return output
    def Run(self,O_list,T_list,net_list = None):
        loss_sum = 0
        OTN_list = [O_list,T_list,net_list]
        for i in range(self.item_num):
            w = self.weight_list[i]
            lf = self.loss_list[i]
            p_list = []
            num = 0
            for p in self.input_list[i]:
                t,n = p
                if n>=len(OTN_list[t]):
                    num = -1
                    break
                p_list.append(OTN_list[t][n])
                num+=1

            if num == 0:
                v = lf()
            elif num ==1:
                v = lf(p_list[0])
            elif num ==2:
                v = lf(p_list[0],p_list[1].detach())
            elif num ==3:
                v = lf(p_list[0],p_list[1].detach(),p_list[2])
            elif num == 4:
                v = lf(p_list[0], p_list[1].detach(), p_list[2],p_list[3])
            elif num == -1:
                v = 0
            else:
                v = lf(p_list)
            loss_sum+=w*v

        self.current_lossv+=loss_sum.data[0]
        return loss_sum

class EmptyVariable(Variable):
    def __init__(self):
        super(EmptyVariable,self).__init__()
        self.data = torch.FloatTensor([0])
    def backward(self, gradient=None, retain_graph=None, create_graph=None, retain_variables=None):
        pass

class EmptyLossFunctionProcess(LossFunctionProcess):
    def __init__(self):
        #super(EmptyLossFunctionProcess,self).__init__()
        pass
    def Run(self,O_list,T_list,net_list = None):
        v= EmptyVariable()
        return v
