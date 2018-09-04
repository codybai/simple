# -*- coding: utf-8 -*-
import sys

path = "./PortraitMattingTest/"
sys.path.insert(0, path)
import os
import base64
import requests
import numpy as np
import cv2
from collections import OrderedDict

import datatransform
data_tranform = datatransform.transform()
class FaceSegmentServer(object):

    def __init__(self, img=None, path=None, encode_type='.jpg'):
        self.portraitUrl = "http://172.16.3.247:8000/segmentation3/humanbody"
        self.skinUrl = "http://172.16.3.247:8000/segmentation3/humanskin"
        self.hairUrl = "http://172.16.3.247:8000/segmentation3/humanhair"
        self.faceUrl = "http://172.18.11.163:7100/api/facial_segment/v20170913"
        self.filepath = path

        if img is not None:
            retval, buffer = cv2.imencode(encode_type, img)
            # Convert to base64 encoding
            self.input_data = {'image_base64': base64.b64encode(buffer)}

        self.info_facepart_range = OrderedDict([
            ('background', (0)),
            ('face_skin', (1)),
            ('left_brow', (2)),
            ('right_brow', (3)),
            ('left_eye', (4)),
            ('right_eye', (5)),
            ('nose', (6)),
            ('mouth', (7)),
            ('left_ear',(10)),
            ('right_ear', (11)),
        ])

    def imageTobuff(self, image, encode_type='.jpg'):
        buffer = None
        try:
            retval, buffer = cv2.imencode(encode_type, image)
            return buffer
        except:
            return buffer

    def decodeBase64(self, segmentationMask_base64):
        nparr = np.fromstring(base64.b64decode(segmentationMask_base64), np.uint8)
        segmentationMask = cv2.imdecode(nparr, 3)
        return segmentationMask

    def getPortrait(self, input_image=None, needMatting=False, needPortrait=False,encode_type='.jpg'):
        portraitMask = None
        mattingMask  = None
        portraitResult = None
        if input_image==None:
            input_data = self.input_data
        else:
            input_data = {"image_base64": data_tranform.ImageEncodeToBase64(input_image, encode_type), 'needMatting':needMatting, 'needPortrait':needPortrait}

        a = requests.post(self.portraitUrl, data=input_data)
        try:
            portraitMask_base64 = a.json()['portraitMask']
            portraitMask = data_tranform.Base64DecodeToImage(portraitMask_base64)


            return portraitMask
        except:
            return portraitMask


    def getSkin(self,input_image=None, encode_type='.jpg'):
        skinMask = None
        if input_image==None:
            input_data = self.input_data
        else:
            input_data = {"image_base64": data_tranform.ImageEncodeToBase64(input_image, encode_type)}

        a = requests.post(self.skinUrl, data=input_data)
        try:
            skinMask_base64 = a.json()['skinMask']
            skinMask = data_tranform.Base64DecodeToImage(skinMask_base64)
            return skinMask
        except:
            return skinMask


    def getHair(self, input_image=None, encode_type='.jpg'):
        hairMask = None
        if input_image==None:
            input_data = self.input_data
        else:
            input_data =  {"image_base64": data_tranform.ImageEncodeToBase64(input_image, encode_type)}

        a = requests.post(self.hairUrl, data=input_data)
        try:
            hairMask_base64 = a.json()['hairMask']
            hairMask = data_tranform.Base64DecodeToImage(hairMask_base64)
            return hairMask
        except:
            return hairMask

    def getWholeFacePart(self, input_image=None):
        url = 'http://172.18.11.163:7100/api/facial_segment/v20170913'
        if input_image is None:
            input_data = self.input_data
        else:
            input_data = input_image

        image_base64 = base64.b64encode(cv2.imencode('.png', input_data)[1])

        r = requests.post(url, data={'image_base64':image_base64}, files={})

        d = r.json()
        if r.status_code == 200:
            img_data = base64.b64decode(d['mask_data'])
            a = np.asarray(bytearray(img_data), dtype=np.uint8)
            mask = cv2.imdecode(a, -1)
            return mask
        else:
            raise RuntimeError(r.content)


    def getFacePart(self, face_part_mask, region, size):
        face_server_foreground = np.zeros(face_part_mask.shape, np.uint8)
        face_server_foreground[face_part_mask == self.info_facepart_range[region]] = 255
        face_server_foreground = cv2.merge([face_server_foreground, face_server_foreground, face_server_foreground])
        face_server_foreground = cv2.resize(face_server_foreground, (size, size), interpolation=cv2.INTER_CUBIC)
        return face_server_foreground

    @staticmethod
    def get_mask(file_path):
        origin = cv2.imread(file_path)
        print 'origin shape:', origin.shape
        url = 'http://172.18.11.163:7100/api/facial_segment/v20170913'
        print "post get hair", file_path
        fake_name = os.urandom(2).encode('hex')
        files = {'imagefile': (
        fake_name + '.jpg', open(file_path, 'rb'))}  # the remote service does not surpport chineses path name
        r = requests.post(url, data={}, files=files)
        d = r.json()
        print d
        if r.status_code == 200:
            img_data = base64.b64decode(d['maskData'])
            a = np.asarray(bytearray(img_data), dtype=np.uint8)
            return cv2.imdecode(a, -1)
        else:
            raise RuntimeError(r.content)

    @staticmethod
    def save_image(path, img):
        data_dir = os.path.dirname(path)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        cv2.imwrite(path, img)


if __name__ == '__main__':
    root = '/root/dl-beauty/data/test_data/DIA_test/test_img_XL/face/train'
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)

        img = cv2.imread(path)

        server = FaceSegmentServer(img, path)
        skin_mask = server.getSkin(img)
        hair_mask = server.getHair(img)
        portrait_mask = server.getPortrait(img)

        face_part_mask = server.getWholeFacePart(img)
        face_server_foreground = server.getFacePart(face_part_mask, 'background', 512)
        left_ear_server_foreground = server.getFacePart(face_part_mask, 'left_ear', 512)

        cv2.imshow('img',  cv2.resize(img, (skin_mask.shape[:2])))
        cv2.imshow('skin', skin_mask)
        cv2.imshow('hair', hair_mask)
        cv2.imshow('portrait', portrait_mask)
        cv2.imshow('face_server_foreground', face_server_foreground)
        cv2.imshow('left_ear_server_foreground', left_ear_server_foreground)

        while cv2.waitKey() != 27:
            pass
