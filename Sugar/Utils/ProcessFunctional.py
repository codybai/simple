import torch
import numpy as np
import cv2
def show_label(ts):
    if len(ts.size())==4:
        _, mask_arg = torch.max(ts, 1)
    else:
        mask_arg = ts
    bc, h, w = mask_arg.size()
    c_bg = torch.clamp(1-torch.abs(mask_arg-0),0,1)
    c_cloth = torch.clamp(1-torch.abs(mask_arg-1),0,1)
    c_skin = torch.clamp(1-torch.abs(mask_arg - 2), 0, 1)
    c_face = torch.clamp(1-torch.abs(mask_arg - 3), 0, 1)
    c_hair = torch.clamp(1-torch.abs(mask_arg - 4), 0, 1)

    r = c_hair+c_face
    g = c_skin+c_face
    b = torch.clamp((c_cloth+c_skin+c_hair)*(1-c_face),0,1)
    rgb = torch.cat([r,g,b],1).float().view(bc,3,h,w)*2-1
    return rgb

def rgb2label(mask_tar):
    r = mask_tar[:, 0:1, :, :]
    g = mask_tar[:, 1:2, :, :]
    b = mask_tar[:, 2:3, :, :]
    hair_ = r * b
    skin_ = b * g
    face_ = (r * g)*(1-b)
    cloth_ = b * (1 - g) * (1 - r)
    bg_ = (1 - r) * (1 - g) * (1 - b)
    mutimask = torch.cat([bg_, cloth_, skin_, face_, hair_], 1)
    _, mask_arg = torch.max(mutimask, 1)
    return mutimask,mask_arg

def rgb2label(mask_tar):
    r = mask_tar[:, 0:1, :, :]
    g = mask_tar[:, 1:2, :, :]
    b = mask_tar[:, 2:3, :, :]
    hair_ = r * b
    skin_ = b * g
    face_ = (r * g)*(1-b)
    cloth_ = b * (1 - g) * (1 - r)
    bg_ = (1 - r) * (1 - g) * (1 - b)
    mutimask = torch.cat([bg_, cloth_, skin_, face_, hair_], 1)
    _, mask_arg = torch.max(mutimask, 1)
    return mutimask,mask_arg

def rgb2hsc(mask_tar):
    r = mask_tar[:, 0:1, :, :]
    g = mask_tar[:, 1:2, :, :]
    b = mask_tar[:, 2:3, :, :]
    hair_ = r * b*(1-g)
    cloth_ = b * (1 - g) * (1 - r)
    bg_ = (1 - r) * (1 - g) * (1 - b)
    skin_ = 1-hair_-cloth_-bg_

    mutimask = torch.cat([bg_, hair_, skin_, cloth_], 1)
    _, mask_arg = torch.max(mutimask, 1)
    return mutimask,mask_arg

def super_label(mask_tar):
    r = mask_tar[:, 0:1, :, :]
    g = mask_tar[:, 1:2, :, :]
    b = mask_tar[:, 2:3, :, :]
    hair_ = r * b *(1-g)
    skin_ = b * g
    face_ = (r * g)*(1-b)
    cloth_ = b * (1 - g) * (1 - r)
    mouth = r*(1-b)*(1-g)
    nose = (1-r)*g*(1-b)
    eye = b*(1-g)*(1-torch.abs(0.5-r))
    brow = (1-r)*(1-torch.abs(0.5-g))*(1-b)
    bg_ = 1-torch.clamp((r+g+b)*2,0,1)
    mutimask = torch.cat([bg_, cloth_, skin_, face_, hair_,mouth,nose,eye,brow], 1)
    _, mask_arg = torch.max(mutimask, 1)
    bg_mask = torch.cat([1-bg_]*3,1)
    return mutimask,mask_arg,bg_mask

def show_super_mask(ts):
    if len(ts.size())==4:
        _, mask_arg = torch.max(ts, 1)
    else:
        mask_arg = ts
    mask_arg = mask_arg.float()
    bc, h, w = mask_arg.size()
    c_bg = torch.clamp(1-torch.abs(mask_arg-0),0,1)
    c_cloth = torch.clamp(1-torch.abs(mask_arg-1),0,1)
    c_skin = torch.clamp(1-torch.abs(mask_arg - 2), 0, 1)
    c_face = torch.clamp(1-torch.abs(mask_arg - 3), 0, 1)
    c_hair = torch.clamp(1-torch.abs(mask_arg - 4), 0, 1)
    c_mouth = torch.clamp(1 - torch.abs(mask_arg - 5), 0, 1)
    c_nose = torch.clamp(1 - torch.abs(mask_arg - 6), 0, 1)
    c_eye = torch.clamp(1 - torch.abs(mask_arg - 7), 0, 1)
    c_brow = torch.clamp(1 - torch.abs(mask_arg - 8), 0, 1)

    r = c_hair+c_face+0.5*c_eye+c_mouth
    g = c_skin+c_face+c_nose+0.5*c_brow
    b = torch.clamp((c_cloth+c_skin+c_hair+c_eye)*(1-c_face),0,1)
    rgb = torch.cat([r,g,b],1).float().view(bc,3,h,w)*2-1
    return rgb

def alpha_bg(mutimask):
    bg_ = mutimask[:, 0:1, :, :]
    cloth_ = mutimask[:, 1:2, :, :]
    skin_ = mutimask[:, 2:3, :, :]
    hair_ = mutimask[:, 4:5, :, :]
    face_ = mutimask[:, 3:4, :, :]
    alpha_mask = torch.cat([1-bg_]*3,1)
    hair_skin_mask = torch.cat([face_+hair_*0.5+skin_*0.1]*3,1)
    Type_tensor =  type(bg_.data)
    r = Type_tensor(bg_.size())
    g = Type_tensor(bg_.size())
    b = Type_tensor(bg_.size())
    r.fill_(np.random.random()*2-1)
    g.fill_(np.random.random()*2-1)
    b.fill_(np.random.random()*2-1)
    bg =type(bg_)(torch.cat([r,g,b],1))
    return alpha_mask,bg.detach(),hair_skin_mask.detach()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)






def alpha_blending(fg,mask,bg):
    b,c,h,w = fg.size()
    mb,mc,mh,mw = mask.size()

    if mc == 1 :
        mask = torch.cat([mask]*c,1)
    output = fg*mask+bg*(1-mask)

    return output

def c4_alpha_blending(output,bg):
    temp = output[:, :3]
    mask_x = torch.cat([(output[:, 3:] + 1) / 2] * 3, 1)
    final = alpha_blending(temp,mask_x,bg)

    return final, mask_x,temp

def show_mask(mask,test=False):
    b,c,h,w = mask.size()
    if c==1:
        Type_tensor = type(mask.data)
        r = Type_tensor(mask.size())
        r.fill_(0)
        r= type(mask)(r)

        if test:
            mask =torch.max(torch.cat([1-mask,mask],1),1)[1].view(b,1,h,w).float()

        mask = torch.cat([r,mask,r],1)

    return mask*2-1

def show_mmc(mask):

    if len(mask.size())==4 :
        _, mask_arg = torch.max(mask, 1)
    else:
        mask_arg = mask

    bc, h,w = mask_arg.size()
    max_ = torch.clamp(1-torch.abs(mask_arg-0),0,1)
    min_ = torch.clamp(1-torch.abs(mask_arg-1),0,1)
    center = torch.clamp(1-torch.abs(mask_arg - 2), 0, 1)

    r = max_
    g = min_
    b = center
    rgb = torch.cat([r,g,b],1).float().view(-1,3,h,w)*2-1
    return rgb

def show_mask_2(mask):
    mask_ = mask[:,1:2,:,:]*2-1
    return torch.cat([mask_]*3,1)

def init_layer(model,layer_index):
    xx_l = []
    print '-----Reset the layer-----'
    for xx in model._modules:
        xx_l.append(xx)

    for i in layer_index:
        #print model._modules[xx_l[i]].weight.data
        #print model._modules[xx_l[i]].bias.data
        model._modules[xx_l[i]].weight.data.normal_(0.0, 0.02)
        model._modules[xx_l[i]].bias.data.fill_(0)

def clip_param(m):
    IntMax = 2 ** 8
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data = ((m.weight.data*IntMax).long()/IntMax).float()
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        m.weight.data = ((m.weight.data*IntMax).long()/IntMax).float()
        m.bias.data = ((m.bias.data*IntMax).long()/IntMax).float()

class AugList():
    def __init__(self):
        self.list = [
            self.light,
            self.blur,
            self.noise
        ]
        self.num = len(self.list)
    def light(self,input):
        yuv = cv2.cvtColor(input,cv2.COLOR_BGR2YUV)
        yuv[:,:,0]=np.clip(yuv[:,:,0]*(np.random.random()*0.3+0.85),0,255).astype(np.uint8)
        output = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
        return output

    def blur(self,input):
        output = cv2.GaussianBlur(input,(3,3),np.random.random())
        return output

    def noise(self,input):
        noise = np.random.random(input.shape)*0.2+0.9
        noise = self.blur(noise)
        output = np.clip(input*noise,0,255).astype(np.uint8)
        return output

    def process(self,input):
        index = int(np.random.random()*10086)%self.num

        return self.list[index](input)

class T_parallel():
    def __init__(self,gpu_id_list):
        self.gpu_id_list = gpu_id_list

    def __call__(self, *args, **kwargs):
        model = args[0]
        note = args[1]
        output = torch.nn.parallel.data_parallel(model,note,self.gpu_id_list)
        return output

def min_filter(image, k_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (k_size, k_size))
    eroded = cv2.erode(image, kernel)
    return eroded

def max_filter(image, k_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (k_size, k_size))
    dilated = cv2.dilate(image, kernel)
    return dilated
def detach_mask(x,mask):
    return x*mask+(x*(1-mask)).detach()

def patch_forward(pos_,m_i,input_ii,pad,pad_size):
    t, b, l, r = pos_  # the part position
    part_i = input_ii[:, :, t:b, l:r]  # input data
    bb, cc, hh, ww = input_ii.size()
    yy = pad_size - t  # crop position
    xx = pad_size - l
    o_i = m_i(part_i)  # output_part
    output_i = pad(o_i + 1)[:, :, yy:yy + hh, xx:xx + ww] - 1  # pad the part to src size
    return output_i

def numpy_super_mask_split(mask):
    r = mask[:, : , 2]
    g = mask[:, : , 1]
    b = mask[:, : , 0]

    r_f = r >250
    g_f = g >250
    b_f = b >250

    r_h = r == 127
    g_h = g == 127

    r_e = r <10
    g_e = g <10
    b_e = b <10



    hair_ = np.bitwise_and(np.bitwise_and(b_f,g_e),r_f)
    face_ = np.bitwise_and(np.bitwise_and(b_e,g_f),r_f)
    skin_ = np.bitwise_and(np.bitwise_and(b_f,g_f),r_e)
    cloth_ = np.bitwise_and(np.bitwise_and(b_f,g_e),r_e)
    brow = np.bitwise_and(np.bitwise_and(b_e, g_h), r_e)
    eye = np.bitwise_and(np.bitwise_and(b_f, g_e), r_h)
    nose = np.bitwise_and(np.bitwise_and(b_e,g_f),r_e)
    mouth = np.bitwise_and(np.bitwise_and(b_e, g_e), r_f)
    body = np.bitwise_not(np.bitwise_and(r_e,np.bitwise_and(g_e,b_e)))
    return body,hair_,face_,skin_,cloth_,brow,eye,nose,mouth

def warp(Image,indexmap):
    imgRotation = torch.nn.functional.grid_sample(
        Image,
        indexmap.transpose(1,2).transpose(2,3))
    return imgRotation

def ExFacePoint(facePoint):
    face_ = facePoint[:32]
    C_point = facePoint[71:72]+ (facePoint[71:72]- facePoint[80:81])*1.5
    A_face = face_[:16][::-1]

    B_face = face_[16:]

    A_face = np.concatenate((A_face[3:], C_point), 0)
    B_face = np.concatenate((B_face[3:], C_point), 0)
    y_list_r = A_face[:,0]
    x_list_r = A_face[:,1]
    coeffs = np.polyfit(x_list_r, y_list_r, 4)
    p_np_A = np.poly1d(coeffs)

    y_list_r = B_face[:, 0]
    x_list_r = B_face[:, 1]
    coeffs = np.polyfit(x_list_r, y_list_r, 4)

    p_np_B = np.poly1d(coeffs)

    step = (C_point[0,1] - A_face[-2,1])//5

    Aex_list = []
    Bex_list = []
    for i in range(5):
        ax = int(A_face[-2,1]+step*i)
        ay = int(p_np_A(ax))

        Aex_list.append([ay,ax])
    Aex_list.reverse()
    step = (C_point[0,1] - B_face[-2, 1]) // 5
    for i in range(5):
        bx = int(B_face[-2, 1] + step * i)
        by = int(p_np_B(bx))

        Bex_list.append([by,bx])

    C_x = int(A_face[-2,1]+step*5+ax+bx)//3
    C_y = int(p_np_A(C_x)+ay+by)//3
    Bex_list.append([C_y, C_x])

    output = np.concatenate((face_,np.array(Bex_list+Aex_list)),0)
    return output