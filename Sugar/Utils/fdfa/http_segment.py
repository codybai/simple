# -*- coding: utf-8 -*-

import os
import mimetypes
import mimetools
import httplib
import base64

"""
Call for the hair mask interface from remote
"""


def http_call_url(imgname, imgpath, ip="172.16.3.32", url = "/seg_hair_upload_json", wSelect="false", socketPort =1052):
    params = encode_multipart_formdata(imgname, imgpath, wSelect)
    headers = {}
    if  params['content_type']:
        headers['Content-Type'] = params['content_type']
    headers['Connection'] = 'Keep-Alive'
    headers['Referer'] = 'http://' + ip + '/'
    conn = httplib.HTTPConnection(ip, socketPort)
    conn.request(method="POST", url=url, body=params['body'], headers=headers)
    resp = conn.getresponse().read()
    return resp


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
    data.append('Content-Disposition: form-data; name="%s"; filename="%s"' % ('imagefile', imgname))
    data.append('Content-Type: %s\r\n' % type )
    # print 'image type:  ' + type
    fr=open(r'%s' % imgpath,'rb')
    data.append(fr.read())
    fr.close()
    # print 'append data ok'
    data.append('--%s--\r\n' % boundary)
    body = CRLF.join(data)
    content_type = 'multipart/form-data; boundary=%s' % boundary
    return {'content_type':content_type, 'body':body}


def get_content_type(filepath):
    return mimetypes.guess_type(filepath)[0] or 'application/octet-stream'


def test_Hair_Segmentation_http_call(srcImageFloder, dstImageFloder):
    headers = {"Content-Type":"application/x-www-form-urlencoded",
               "Connection":"Keep-Alive"};
    #  define the header
    if not os.path.exists(dstImageFloder):
        os.makedirs(dstImageFloder)

    ip = "172.16.3.32"
    url = "/seg_hair_upload_json"
    socketPort = 1052
    imageNameList = os.listdir(srcImageFloder)
    # for index in range(0, len(imageNameList)):
    #     print index
    #     imageName = imageNameList[index]
    #     imagePath = os.path.join(srcImageFloder, imageName)
    for name in imageNameList:
        # print index
        # imageName = imageNameList[index]
        imagePath = os.path.join(srcImageFloder, name)
        imageName = name.split('.jpg')[0]
        result = http_call_url(name, imagePath, ip="172.16.3.32", url="/seg_hair_upload_json",
                               wSelect='true', socketPort=socketPort)
        evalResult= eval(result)
        if evalResult.has_key("imageData"):  #get image data from json
            imgData = base64.b64decode(evalResult['imageData'])
            # leniyimg = open(os.path.join(dstImageFloder,str(index)+'_imgout.png'),'wb')
            leniyimg = open(os.path.join(dstImageFloder, imageName+'.png'), 'wb')
            leniyimg.write(imgData)
            leniyimg.close()


def test_hair_segmentation_http_call_single(filename, filename_, UPLOAD_FOLDER_MASK):
    headers = {"Content-Type":"application/x-www-form-urlencoded",
               "Connection":"Keep-Alive"}
    #  define the header
    ip = "172.16.3.32"
    url = "/seg_hair_upload_json"
    socketPort = 1052
    name = filename.split('.jpg')[0]
    result = http_call_url(filename, filename_, ip="172.16.3.32", url = "/seg_hair_upload_json", wSelect = 'true', socketPort = socketPort)
    evalResult= eval(result)
    if evalResult.has_key("imageData"):  #get image data from json
        imgData = base64.b64decode(evalResult['imageData'])
        # leniyimg = open(os.path.join(dstImageFloder,str(index)+'_imgout.png'),'wb')
        leniyimg = open(os.path.join(UPLOAD_FOLDER_MASK, name+'mask.png'), 'wb')
        leniyimg.write(imgData)
        leniyimg.close()
        mask_path = os.path.join(UPLOAD_FOLDER_MASK, name+'mask.png')
        return mask_path
    return None


def test_skin_segmentation_http_call_single(filename, filename_, UPLOAD_FOLDER_MASK):
    headers = {"Content-Type":"application/x-www-form-urlencoded",
               "Connection":"Keep-Alive"}
    #  define the header
    ip = "172.16.3.32"
    url = "/seg_hair_upload_json"
    socketPort = 1060
    name = filename.split('.jpg')[0]
    result = http_call_url(filename, filename_, ip="172.16.3.32", url="/seg_skin_upload_json", wSelect = 'true', socketPort=socketPort)
    evalResult= eval(result)
    if evalResult.has_key("imageData"):  #get image data from json
        imgData = base64.b64decode(evalResult['imageData'])
        # leniyimg = open(os.path.join(dstImageFloder,str(index)+'_imgout.png'),'wb')
        leniyimg = open(os.path.join(UPLOAD_FOLDER_MASK, name+'skin.png'), 'wb')
        leniyimg.write(imgData)
        leniyimg.close()
        mask_path = os.path.join(UPLOAD_FOLDER_MASK, name+'skin.png')
        return mask_path
    return None


def test_portrait_segmentation_http_call_single(filename, filename_, UPLOAD_FOLDER_MASK):
    headers = {"Content-Type":"application/x-www-form-urlencoded",
               "Connection":"Keep-Alive"}
    #  define the header
    ip = "172.16.3.32"
    url = "/seg_hair_upload_json"
    socketPort = 1061
    name = filename.split('.jpg')[0]
    result = http_call_url(filename, filename_, ip="172.16.3.32", url = "/portrait_matting_upload_json", wSelect = 'true', socketPort = socketPort)
    evalResult= eval(result)
    if evalResult.has_key("imageData"):  #get image data from json
        imgData = base64.b64decode(evalResult['imageData'])
        # leniyimg = open(os.path.join(dstImageFloder,str(index)+'_imgout.png'),'wb')
        leniyimg = open(os.path.join(UPLOAD_FOLDER_MASK, name+'mask.png'), 'wb')
        leniyimg.write(imgData)
        leniyimg.close()
        mask_path = os.path.join(UPLOAD_FOLDER_MASK, name+'mask.png')
        return mask_path
    return None
