import cv2
import numpy as np
from Utils.fdfa.face_segment import FaceSegment
import torch
def GetRotateMat(Apoint,Bpoint,vector):
    l = (vector ** 2).sum() ** 0.5
    sin_ = vector[1]/l
    cos_ = vector[0]/l
    mat = [
        [cos_,-sin_],
        [sin_,cos_],
        [Apoint[0],Apoint[1]],
        [Bpoint[0],Bpoint[1]]
    ]

    mat_I = [
        [cos_,sin_],
        [-sin_,cos_],
        [Bpoint[0], Bpoint[1]],
        [Apoint[0], Apoint[1]]
    ]
    return np.array(mat),np.array(mat_I)

def use_mat(X_map_,Y_map_,mat):
    A_x = mat[3][0]
    A_y = mat[3][1]

    B_x = mat[2][0]
    B_y = mat[2][1]

    X_map = X_map_ - A_x
    Y_map = Y_map_ - A_y
    X = X_map * mat[0,0] + Y_map * mat[0,1]
    Y = X_map * mat[1,0] + Y_map * mat[1,1]
    X += B_x
    Y += B_y
    return X,Y

def one_face(Image,sem,fp):
    xl = fp[:, 0:1]
    yl = fp[:, 1:2]
    xmin = xl.min()
    xmax = xl.max()
    ymin = yl.min()
    ymax = yl.max()
    c_x = (fp[59][0]+fp[69][0])//2
    c_y = (fp[59][1]+fp[69][1])//2
    half_arr =np.array([
        abs(c_x - xmin),
        abs(c_x - xmax),
        abs(c_y - ymin),
        abs(c_y - ymax)
    ])
    p_size = half_arr.max()*1.5
    pp_size = int(p_size*2)
    vector = fp[69] - fp[59]
    mat, mat_I = GetRotateMat((c_x, c_y),(p_size, p_size),vector)
    X_map_, Y_map_ = np.meshgrid(range(pp_size), range(pp_size))

    X_ ,Y_ = use_mat(X_map_, Y_map_,mat)

    imgRotation = cv2.remap(
        Image,
        X_.astype(np.float32),
        Y_.astype(np.float32),
        cv2.INTER_LINEAR)
    semRotation = cv2.remap(
        sem,
        X_.astype(np.float32),
        Y_.astype(np.float32),
        cv2.INTER_LINEAR)
    warp_xl,warp_yl = use_mat(xl,yl, mat_I)
    warp_xy = np.concatenate([warp_xl,warp_yl],1)
    return imgRotation,semRotation,warp_xy,mat_I



sum_list = [
    0,0,0,0,0,0,0,1
]

def face_position(face,h,w):
    u_lefteye = face[51:59]
    u_righteye = face[61:69]
    u_nose = face[71:86]
    u_mouth = face[86:106]
    u_face = face[2:32:2]
    u_leftbrow = face[33:42]
    u_rightbrow = face[42:51]
    point_list = [
        np.concatenate((u_face, u_leftbrow - (0, 128)), 0),
        u_nose,
        u_mouth,
        u_lefteye,
        u_righteye,
        u_leftbrow,
        u_rightbrow
    ]
    tblr_list = []
    padding = 32
    for i,pl in enumerate(point_list):
        xl = pl[:, 0]
        yl = pl[:, 1]
        t = np.clip(yl.min() - padding,0,h)
        b = np.clip(yl.max() + padding,0,h)
        l = np.clip(xl.min() - padding,0,w)
        r = np.clip(xl.max() + padding,0,w)
        #area = (b-t)*(l-r)
        #sum_list[i]+=area
        #print sum_list[i]//sum_list[-1]
        tblr_list.append([t, b, l, r])
    #sum_list[-1]+=1

    return tblr_list

def ReRotate(src,Image,matI):
    hh,ww = src.shape[:2]
    X_map_, Y_map_ = np.meshgrid(range(ww), range(hh))
    grid_x ,grid_y = use_mat(X_map_,Y_map_,matI)
    imgRotation = cv2.remap(
        Image,
        grid_x.astype(np.float32),
        grid_y.astype(np.float32),
        cv2.INTER_LINEAR)

    return imgRotation
from torch.autograd import Variable
def ReRotateTorch(src,Image,matI):
    bb,cc,hh,ww = src.size()
    fb, fc,fh, fw = Image.size()
    theta = Variable(torch.FloatTensor(
        np.array([[[1,0, 0],
        [0, 1, 0]]])
    ).cuda())
    XY = torch.nn.functional.affine_grid(
        theta,
        src.size()
    ).cuda()
    X_map_ = (XY[:, :, :, 0:1] + 1) * 0.5 * ww
    Y_map_ = (XY[:, :, :, 1:2] + 1) * 0.5 * hh

    grid_x ,grid_y = use_mat(X_map_,Y_map_,matI)

    X_map_ = grid_x/fw*2-1
    Y_map_ = grid_y/fh*2-1

    XY = torch.cat([X_map_,Y_map_],3)

    imgRotation = torch.nn.functional.grid_sample(
        Image+1,
        XY)-1
    return imgRotation

def FDFA(Image):
    faceseg = FaceSegment()
    faceseg.set_input(Image)
    facePoint = faceseg.get_faces_key_points()
    if facePoint is None:
        return np.array([[0]]), Image,[Image],np.array([[0]])

    face_tblr = []
    face_list = []

    matI_list = None
    sem_mask = faceseg.get_sem_mask()
    for i in range(len(facePoint)):
        face = facePoint[i]
        warp_face,warp_seg,face_,mat_I = one_face(Image,sem_mask,face)
        hh,ww = warp_face.shape[:2]
        face_tblr+=face_position(face_,hh,ww)
        face_list.append(warp_face)
        face_list.append(warp_seg)
        if matI_list is None :
            matI_list = mat_I
        else:
            matI_list = np.concatenate([matI_list,mat_I],0)

    array_pos = np.array(face_tblr)
    return array_pos,sem_mask,face_list,matI_list
