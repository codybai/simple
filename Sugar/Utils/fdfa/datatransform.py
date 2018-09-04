import cv2
import base64
import numpy as np
class transform(object):
    def __init__(self, ):
        pass

    def ImageToBuff(self, image, encode_type='.jpg'):
        buffer =None
        try:
            retval, buffer = cv2.imencode(encode_type, image)
            return buffer

        except:
            return buffer

    def BuffToImage(self, buff):
        image = None
        try:
            nparr = np.fromstring(buff, np.uint8)
            image = cv2.imdecode(nparr, 3)
            return image

        except:
            return image

    def Base64DecodeToImage(self, segmentationMask_base64):
        segmentationMask = None
        nparr = np.fromstring(base64.b64decode(segmentationMask_base64), np.uint8)
        segmentationMask = cv2.imdecode(nparr, 3)

        return segmentationMask

    def ImageEncodeToBase64(self, image, encode_type='.jpg'):
        buffer = self.ImageToBuff(image, encode_type)
        base64_data = base64.b64encode(buffer)

        return base64_data

    def BuffEncodeToBase64(self, buff):

        return base64.b64encode(buff)
