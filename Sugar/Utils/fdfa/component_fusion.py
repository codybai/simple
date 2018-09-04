# -*- coding: utf-8 -*-

import cv2
import numpy as np
from collections import Iterable

from fdfa.face_segment import FaceSegment, WarpedFaceSegment


class ComponentFusion(object):
    def __init__(self, resolution=512, crop=True):
        super(ComponentFusion, self).__init__()

        if not isinstance(resolution, Iterable):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = resolution

        self.crop = crop
        self.face_segment = WarpedFaceSegment(resolution) if crop else FaceSegment(resolution)

    def set_source(self, source):
        self.source = source
        self.face_segment.set_input(source)
        self.key_points = None

    def get_face_key_points(self):
        if self.crop:
            self.source, self.key_points = self.face_segment.get_face_key_points()
            return self.source, self.key_points
        else:
            self.key_points = self.face_segment.get_face_key_points()
            return self.key_points

    def get_components(self):
        assert self.key_points is not None, 'method get_face_key_points should be called first'

        self.src_components, self.com_boxes, self.com_masks, self.com_pts = self.face_segment.get_facial_components()
        return self.src_components, self.com_boxes, self.com_pts

    def set_target(self, target):
        self.tgt_sem = target[0]
        self.tgt_components = target[1:]

    def get_fusion_result(self):
        self.fusion = self.tgt_sem.copy()
        self.fuse_facial_components()
        return self.fusion

    def fuse_facial_components(self):
        # nose should be fused before eyes
        order = [0, 1, 4, 2, 3, 5]
        reordered_coms = [self.tgt_components[i] for i in order]
        reordered_com_boxes = [self.com_boxes[i] for i in order]
        reordered_com_masks = [self.com_masks[i] for i in order]
        naive_fusion = self.fusion.copy()
        for label, com, box, mask in zip(order, reordered_coms, reordered_com_boxes, reordered_com_masks):
            x, y, w, h = box

            if com.shape[0] != h or com.shape[1] != w:
                com = cv2.resize(com, (w, h))

            if label in [2, 3]:
                niter = 8 * max(1, self.resolution[1]/512)
            else:
                niter = 6 * max(1, self.resolution[1]/512)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=niter)
            blur_mask = cv2.blur(mask, (niter, niter), borderType=cv2.BORDER_CONSTANT) / 255.0

            fg = com * blur_mask
            bg = naive_fusion[y:y+h, x:x+w] * (1.0-blur_mask)

            naive_fusion[y:y + h, x:x + w] = bg + fg

        # eliminate facial components region outside face
        face_mask = self.face_segment.get_face_mask()
        self.fusion = np.bitwise_and(self.fusion, np.bitwise_not(face_mask)) + np.bitwise_and(naive_fusion, face_mask)
