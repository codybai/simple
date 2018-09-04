# -*- coding: utf-8 -*-
import os
import shutil
import string
import cv2
import MT2Python
import sys,os,urllib2
import mimetypes
import mimetools
import httplib,urllib
import base64


def memory_solve(face_list_source):
    try:
        face_list = []
        face_num = len(face_list_source)
        for i in range(0, face_num):
            face_list.append(face_list_source[i][0])
            face_list.append(face_list_source[i][1])
            face_list.append(face_list_source[i][2])
            face_list.append(face_list_source[i][3])
        del face_list_source
        return face_list, face_num
    except Exception as error:
        # print 'cannot get the face'
        print 'this is not the needed file'


def http_call_url(imgname, imgpath, ip="192.168.43.193", url = "/seg_hair_upload_json", wSelect="false", socketPort =1052):
    params = encode_multipart_formdata(imgname, imgpath, wSelect)
    headers ={}
    if  params['content_type']:
        headers['Content-Type']=params['content_type']
    headers['Connection']='Keep-Alive'
    headers['Referer']='http://'+ ip +'/'
    conn = httplib.HTTPConnection(ip, socketPort)
    conn.request(method="POST", url=url,body=params['body'],headers=headers)
    resp = conn.getresponse().read()
    return resp


def get_content_type(filepath):
    return mimetypes.guess_type(filepath)[0] or 'application/octet-stream'


def encode_multipart_formdata(imgname, imgpath, wSelect="false"):
    boundary = mimetools.choose_boundary()
    CRLF = '\r\n'
    data = []
    type = get_content_type(imgpath)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"' % 'wSelect')
    data.append('')
    data.append(wSelect)

    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"; filename="%s"' % ('imagefile',imgname))
    data.append('Content-Type: %s\r\n' % type )
    print 'image type:  ' + type
    fr=open(r'%s' % imgpath,'rb')
    data.append(fr.read() )
    fr.close()
    print 'append data ok'
    data.append('--%s--\r\n' % boundary)
    body = CRLF.join(data)
    content_type = 'multipart/form-data; boundary=%s' % boundary
    return {'content_type':content_type,'body':body}


def test_skin_segmentation_http_call_single(filename, filename_, UPLOAD_FOLDER_MASK):
    headers = {"Content-Type":"application/x-www-form-urlencoded",
               "Connection":"Keep-Alive"}
    #  define the header
    ip = "172.16.3.32"
    url = "/seg_hair_upload_json"
    socketPort = 1060
    name = filename_.split('.jpg')[0]
    result = http_call_url(filename_, filename, ip="172.16.3.32", url="/seg_skin_upload_json", wSelect = 'true', socketPort=socketPort)
    evalResult= eval(result)
    if evalResult.has_key("imageData"):  #get image data from json
        imgData = base64.b64decode(evalResult['imageData'])
        # leniyimg = open(os.path.join(dstImageFloder,str(index)+'_imgout.png'),'wb')
        leniyimg = open(os.path.join(UPLOAD_FOLDER_MASK, name+'skin.png'), 'wb')
        leniyimg.write(imgData)
        leniyimg.close()
        mask_path = os.path.join(UPLOAD_FOLDER_MASK, name+'skin.png')
        return mask_path


def face_detect(img_src):

    """
    # return face_list[0] 裁剪的人脸图像， face_list[1] FA得到的人脸点， face_list[2] FD得到的人脸矩形框，
    # return face_list[3] Pose Estimation得到的人脸点, 点数为6, 姿态的旋转角度(3 x 1),姿态的位移向量(3 x 1)
    """
    FaceSize = 512
    afExtentParam = [0.6, 0.0, 0.0, 0.0, 0.3]
    FDScore = 0
    FDScale = 0
    BorderMode = MT2Python.MT_BORDER_CONSTANT
    face_list, fd_t0, fa_t1, warp_t2 = MT2Python.FDFA.DetectWarpedFaceImages_s(img_src, afExtentParam, FaceSize, FDScore, nBorderMode=BorderMode)
    face_list, _ = memory_solve(face_list)
    return face_list


if __name__ == '__main__':
    img_path = '/home/wenyangming/FaceSearch2/1a.jpg'
    face_list = face_detect(cv2.imread(img_path))

    # face_list[0] -> face
    # face_list[1] -> fa_points (before warp 0-212 after 212-414)
    filename = '/home/wenyangming/FaceSearch2/1a_face.jpg'
    filename_ = '1a_face.jpg'
    cv2.imwrite(filename, face_list[0])
    UPLOAD_FOLDER_MASK = '/home/wenyangming/FaceSearch2/'
    face_skin = test_skin_segmentation_http_call_single(filename, filename_, UPLOAD_FOLDER_MASK)
    print 'success'