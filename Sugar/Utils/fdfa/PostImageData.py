#!/usr/bin/python
# -*-coding:utf-8-*-


import sys, os, urllib2
import mimetypes
import mimetools
import httplib, urllib
import base64


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
    data.append('Content-Disposition: form-data; name="%s"; filename="%s"' % ('imagefile', imgname))
    data.append('Content-Type: %s\r\n' % type)
    print 'image type:  ' + type
    fr = open(r'%s' % imgpath, 'rb')
    data.append(fr.read())
    fr.close()
    print 'append data ok'
    data.append('--%s--\r\n' % boundary)
    body = CRLF.join(data)
    content_type = 'multipart/form-data; boundary=%s' % boundary
    return {'content_type': content_type, 'body': body}


def http_call_url(imgname, imgpath, ip="192.168.43.193", url="/seg_hair_upload_json", wSelect="false", socketPort=1052):
    params = encode_multipart_formdata(imgname, imgpath, wSelect)
    headers = {}
    if params['content_type']:
        headers['Content-Type'] = params['content_type']
    headers['Connection'] = 'Keep-Alive'
    headers['Referer'] = 'http://' + ip + '/'
    conn = httplib.HTTPConnection(ip, socketPort)
    conn.request(method="POST", url=url, body=params['body'], headers=headers)
    resp = conn.getresponse().read()
    return resp


if __name__ == "__main__":
    headers = {"Content-Type": "application/x-www-form-urlencoded",
               "Connection": "Keep-Alive"};  # define the header

    srcImageFloder = "/home/meitu/Git/hairsegwebdemo/t"
    # srcImageFloder = "/home/meitu/test_hwd_png"
    dstImageFloder = "/home/meitu/Git/hairsegwebdemo/d"
    if not os.path.exists(dstImageFloder):
        os.makedirs(dstImageFloder)

    ip = "192.168.43.193"
    url = "/seg_hair_upload_json"
    socketPort = 1052

    imageNameList = os.listdir(srcImageFloder)
    for index in range(0, len(imageNameList)):
        print index
        imageName = imageNameList[index]
        imagePath = os.path.join(srcImageFloder, imageName)

        result = http_call_url(imageName, imagePath, url, wSelect='true', socketPort=socketPort)
        evalResult = eval(result)
        if evalResult.has_key("imageData"):  # get image data from json
            imgData = base64.b64decode(evalResult['imageData'])
            leniyimg = open(os.path.join(dstImageFloder, str(index) + '_imgout.png'), 'wb')
            leniyimg.write(imgData)
            leniyimg.close()
