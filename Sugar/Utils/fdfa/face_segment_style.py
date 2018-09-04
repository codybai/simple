# -*- coding: utf-8 -*-

import numpy as np
import cv2

from fdfa.face_segment import FaceSegment


class FaceSegmentStyle(FaceSegment):
    def __init__(self, resolution=(960, 1280), crop_size='XXL'):
        super(FaceSegmentStyle, self).__init__(resolution=resolution, crop_size=crop_size)

        assert self.resolution[0] <= self.resolution[1], 'crop width must be equal to or less than crop height'
        assert (self.resolution[1]-self.resolution[0]) % 2 == 0, 'is (height - width) % 2 == 0 satisfied?'

        self.name = 'FaceSegmentStyle'
        self.is_A = False
        self.border_value = (255, 255, 255)

    def get_face_key_points(self, fa_points=None):
        afExtentParam = self.afExtentParams[self.crop_size]
        if fa_points is None:
            face_list = self.get_face_list(self.ori_img,
                                           afExtentParam,
                                           face_size=-1,
                                           border_fill=self.border_value,
                                           warp_flag=0)
        else:
            face_list = self.get_face_list_by_fa_points(self.ori_img,
                                                        afExtentParam,
                                                        fa_points.reshape(-1),
                                                        face_size=-1,
                                                        border_fill=self.border_value,
                                                        warp_flag=0)
        self.reserved_fdfa_face_list = face_list

        if len(face_list) == 0:
            return None

        face_list = self._rescale_face_list(face_list[0])
        self.face_list = face_list

        num_key_points = len(face_list[1]) / 4
        self.key_points = face_list[1][2*num_key_points:].reshape(-1, 2)
        # TODO(ruanshihai): only support single person case
        self.multi_face_key_points = [self.key_points]
        self.face = face_list[0][:, :, 0:3].copy()
        self.valid_mask = np.repeat(face_list[0][:, :, 3:4], 3, axis=2)
        self.valid_rect = face_list[2][26:30].astype(np.int)

        return self.face, self.key_points

    def _rescale_face_list(self, face_list):
        scale_face_list = list(face_list)

        h0, w0 = scale_face_list[0].shape[0:2]
        w, h = self.resolution
        scale = float(h) / h0
        offset = (h-w) / 2

        # rescale and crop face to desired resolution
        face = cv2.resize(scale_face_list[0], (h, h), interpolation=cv2.INTER_AREA)
        scale_face_list[0] = face[:, offset:-offset, :].copy()

        # correct facial key points to new scale
        num_key_points = len(scale_face_list[1]) / 4
        for i in range(num_key_points):
            scale_face_list[1][2*num_key_points+2*i] *= scale
            scale_face_list[1][2*num_key_points+2*i] -= offset
            scale_face_list[1][2*num_key_points+2*i+1] *= scale

        # affine matrix remains unchanged
        scale_face_list[2][18:24] = scale_face_list[2][18:24]

        return scale_face_list

    def _get_ground_truth_sem_mask(self, img_mask):
        sem_mask = self.ref_crop_another_img(img_mask, border_value=self.mask_colors['bg'])
        return sem_mask

    def ref_crop_another_img(self, img, border_value=None):
        if border_value is None:
            border_value = self.border_value

        warp_mat = np.array(self.face_list[2][18:24]).reshape(2, 3)
        w0, h0 = int(self.face_list[2][16]), int(self.face_list[2][17])
        w, h = self.resolution
        offset = (h-w) / 2

        cropped_img = cv2.warpAffine(img, warp_mat, (w0, h0), borderValue=border_value)
        cropped_img = cv2.resize(cropped_img, (h, h), interpolation=cv2.INTER_AREA)
        cropped_img = cropped_img[:, offset:-offset, :].copy()

        return cropped_img
