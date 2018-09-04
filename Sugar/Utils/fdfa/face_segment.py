# -*- coding: utf-8 -*-

import copy
import numpy as np
import cv2
from collections import OrderedDict
from collections import Iterable
import shutil
import os
if os.path.exists('/usr/lib/libMeituFDFA.so') == False:
    shutil.copy('./Utils/fdfa/libMeituFDFA.so', '/usr/lib/libMeituFDFA.so')
if os.path.exists('/usr/lib/libyuv.so') == False:
    shutil.copy('./Utils/fdfa/libyuv.so', '/usr/lib/libyuv.so')

import MT2Python
from ..fdfa.face_segment_server import FaceSegmentServer



def pick_mask_by_color(img, color, thresh=50):
    if img.shape[2] == 3:
        lowerb = (color[0]-thresh, color[1]-thresh, color[2]-thresh)
        upperb = (color[0]+thresh, color[1]+thresh, color[2]+thresh)
        mask = cv2.inRange(img, lowerb, upperb)
        return np.stack([mask] * 3, axis=2)
    else:
        return cv2.inRange(img, color-thresh, color+thresh)


class FaceSegment(object):
    info_kpts_range = OrderedDict([
        ('face', (0, 32)),
        ('left_brow', (33, 41)),
        ('right_brow', (42, 50)),
        ('left_eye', (51, 60)),
        ('right_eye', (61, 70)),
        ('nose', (71, 85)),
        ('mouth', (86, 105))
    ])

    # component_name: key_points_range, extend
    info_components = OrderedDict([
        ('left_brow', [(33, 41), (1.5, 2.0)]),
        ('right_brow', [(42, 50), (1.5, 2.0)]),
        ('left_eye', [(51, 60), (2.0, 2.5)]),
        ('right_eye', [(61, 70), (2.0, 2.5)]),
        ('nose', [(71, 85), (1.5, 1.5)]),
        ('mouth', [(86, 105), (1.5, 2.0)])
    ])

    mask_colors = {
        'hair': (255, 0, 255),
        'face': (0, 255, 255),
        'skin': (255, 255, 0),
        'cloth': (255, 0, 0),

        'brow': (0, 127, 0),
        'eye': (255, 0, 127),
        'nose': (0, 255, 0),
        'mouth': (0, 0, 255),
        'tooth': (0, 0, 127),

        'bg': (0, 0, 0),
        'others': (255, 255, 255),
    }

    def __init__(self, resolution=64, crop_size='S'):
        super(FaceSegment, self).__init__()
        self.name = 'FaceSegment'

        if not isinstance(resolution, Iterable):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = resolution
        self.crop_size = crop_size
        self.is_A = True
        self.border_value = (0, 0, 0)

        self.afExtentParams = {
            'S': [0.0, 0.0, 0.0, 0.0, 0.0],
            'M': [0.6, 0.0, 0.0, 0.0, 0.3],
            'L': [1.0, 0.0, 0.6, 0.0, 0.0],
            'XL': [1.0, 0.0, 1.2, 0.0, 0.0],
            'XXL': [1.1, 0.0, 1.6, 0.0, 0.0],
            'XL2': [1.1, 0.0, 1.6, 0.0, 0.0],
            'XL3': [1.2, 0.0, 2.0, 0.0, 0.0],
        }

    def set_input(self, img):
        self.ori_img = img
        self.face = None
        self.key_points = None
        self.multi_face_key_points = None
        self.soft_sem_mask = None
        self.sem_mask = None
        self.super_mask = None

    def get_face_list(self, img, afExtentParam, face_size=-1, fd_score=0, fa_score=0.5,
                      interplation=MT2Python.MT_INTER_LINEAR,
                      border_mode=MT2Python.MT_BORDER_CONSTANT, border_fill=0, warp_flag=0):
        face_list = MT2Python.FDFA.DetectWarpedFaceImages(img, afExtentParam, face_size=face_size,
                fd_score=fd_score, fa_score=fa_score, interplation=interplation, border_mode=border_mode,
                border_fill=border_fill, warp_flag=warp_flag)
        face_list = self._copy_to_avoid_memory_leaks(face_list)
        return face_list

    def get_face_list_by_fa_points(self, img, afExtentParam, fa_points, fore_head_dir_x=0.0,
                                   fore_head_dir_y=0.0, face_size=-1, interplation=MT2Python.MT_INTER_LINEAR,
                                   border_mode=MT2Python.MT_BORDER_CONSTANT, border_fill=0, warp_flag=0):
        face_list = MT2Python.FDFA.DetectWarpedFaceImagesByFAPoints(img, afExtentParam, fa_points,
                fore_head_dir_x, fore_head_dir_y, face_size=face_size, interplation=interplation,
                border_mode=border_mode, border_fill=border_fill, warp_flag=warp_flag)
        face_list = self._copy_to_avoid_memory_leaks([face_list])
        return face_list

    @staticmethod
    def _copy_to_avoid_memory_leaks(face_list):
        # TODO(ruanshihai): a bug about MT2Python? Avoid memory leaks.
        copied_face_list = copy.deepcopy(face_list)
        for i in range(len(face_list), 0, -1):
            del face_list[i-1]
        return copied_face_list

    def get_face_key_points(self):
        # TODO(ruanshihai): there exists redundant calculation
        # only facial key points are needed, thus make these arguments for fast calculation
        afExtentParam = self.afExtentParams[self.crop_size]
        face_list = self.get_face_list(self.ori_img,
                                       afExtentParam,
                                       face_size=self.resolution,
                                       border_fill=self.border_value,
                                       warp_flag=0)
        self.reserved_fdfa_face_list = face_list

        if len(face_list) == 0:
            return None

        face_list = face_list[0]

        num_key_points = len(face_list[1]) / 4
        self.key_points = face_list[1][0:2*num_key_points].reshape(-1, 2)
        self.multi_face_key_points = [fl[1][0:2*num_key_points].reshape(-1, 2) for fl in self.reserved_fdfa_face_list]
        self.face = self.ori_img

        return self.key_points

    def get_faces_key_points(self):
        # TODO(ruanshihai): there exists redundant calculation
        # only facial key points are needed, thus make these arguments for fast calculation
        afExtentParam = self.afExtentParams[self.crop_size]
        face_list = self.get_face_list(self.ori_img,
                                       afExtentParam,
                                       face_size=self.resolution,
                                       border_fill=self.border_value,
                                       warp_flag=0)
        self.reserved_fdfa_face_list = face_list

        if len(face_list) == 0:
            return None
        self.face = self.ori_img
        key_list = []
        for i in range(len(face_list)):
            face_ = face_list[i]

            num_key_points = len(face_[1]) / 4
            key_points = face_[1][0:2*num_key_points].reshape(-1, 2)
            key_list.append(key_points)
            self.key_points = key_points
            self.multi_face_key_points = [fl[1][0:2 * num_key_points].reshape(-1, 2) for fl in
                                          self.reserved_fdfa_face_list]

        return key_list

    def set_face_key_points(self, multi_face_fa_points):
        self.key_points = multi_face_fa_points[0]
        self.multi_face_key_points = multi_face_fa_points
        self.face = self.ori_img

    def get_facial_components(self):
        assert self.key_points is not None, 'face key points is not available now'

        self.components = []
        self.com_boxes = []
        self.com_masks = []
        self.com_pts = []
        for k, v in self.info_components.items():
            com, box, mask, pts = self._crop_component(self.face, self.key_points[v[0][0]:v[0][1] + 1], extend=v[1])
            self.components.append(com)
            self.com_boxes.append(box)
            self.com_masks.append(mask)
            self.com_pts.append(pts)
        return self.components, self.com_boxes, self.com_masks, self.com_pts

    @staticmethod
    def _crop_component(img, contour, extend=(1.0, 1.0)):
        # TODO(ruanshihai): unit should be self-adapting for different cases
        unit = 16 * 2
        x0, y0, w0, h0 = cv2.boundingRect(contour[:, np.newaxis, :])

        # extend the cropping window (width and height must be divided exactly by 16 due to INV_VGG)
        cx, cy = x0 + w0 / 2, y0 + h0 / 2
        w, h = int(extend[0] * w0 + 0.01), int(extend[1] * h0 + 0.01)
        w, h = (w + unit - 1) / unit * unit, (h + unit - 1) / unit * unit

        # the extended window must be not beyond the image region
        h_, w_ = img.shape[0:2]
        w, h = min(w, 2 * cx, 2 * (w_ - cx - 1)), min(h, 2 * cy, 2 * (h_ - cy - 1))
        # TODO(ruanshihai): need to handle the very rare cases of small (w<=16 or h<=16) cropping windows near image border
        w, h = w / unit * unit, h / unit * unit

        x, y = max(0, cx - w / 2), max(0, cy - h / 2)

        mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask_pts = np.array([[v[0] - x, v[1] - y] for v in contour]).astype(np.int32)
        mask_pts = cv2.convexHull(mask_pts)
        cv2.fillConvexPoly(mask, mask_pts, (255, 255, 255))

        component = img[y:y + h, x:x + w, :].copy()

        # # mask out background
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # dmask = cv2.dilate(mask, kernel, iterations=4)
        # component = cv2.bitwise_and(component, dmask)

        abs_pts = contour - (x, y)

        return component, (x, y, w, h), mask, abs_pts

    def get_soft_sem_mask(self):
        """
        3 soft masks generated by segmentation networks: hair, skin, cloth (portrait)
        """
        if self.soft_sem_mask is None:
            self._generate_masks()
            self.soft_sem_mask = np.stack([self.soft_portrait_mask[:, :, 0],
                                           self.soft_skin_mask[:, :, 0],
                                           self.soft_hair_mask[:, :, 0]], axis=2)

        return self.soft_sem_mask

    def _generate_masks(self):
        assert self.key_points is not None, 'face key points is not available now'
        server = FaceSegmentServer(self.face)

        # hair mask
        mask = server.getHair()
        if mask is None:
            mask = np.zeros(self.face.shape, np.uint8)
        else:
            mask = cv2.resize(mask, (self.face.shape[1], self.face.shape[0]), interpolation=cv2.INTER_CUBIC)
            mask = np.bitwise_not(mask)
        self.soft_hair_mask = mask

        # skin mask
        mask = server.getSkin()
        if mask is None:
            mask = np.zeros(self.face.shape, np.uint8)
        else:
            mask = cv2.resize(mask, (self.face.shape[1], self.face.shape[0]), interpolation=cv2.INTER_CUBIC)
            mask = np.bitwise_not(mask)
        self.soft_skin_mask = mask

        # TODO(ruanshihai): adopt more elegant solution
        # fill holes of eye brows, eyes, nose and mouth
        #self.soft_skin_mask = self._fill_facial_component_holes()

        # portrait mask
        mask = server.getPortrait(needMatting=True)
        if mask is None:
            mask = np.zeros(self.face.shape, np.uint8)
        else:
            mask = cv2.resize(mask, (self.face.shape[1], self.face.shape[0]), interpolation=cv2.INTER_CUBIC)
            mask = np.bitwise_not(mask)
        self.soft_portrait_mask = mask

    def _fill_facial_component_holes(self):
        h, w = self.soft_skin_mask.shape[0:2]
        hole_mask = np.zeros(self.soft_skin_mask.shape, dtype=np.uint8)

        for face_kpts in self.multi_face_key_points:
            # chin points and shifted top eye-brow points
            padding = 8 * max(1, h / 512)
            chin_pts = face_kpts[range(0, 33, 1)]
            top_left_brow_pts = face_kpts[range(37, 32, -1)] + (0, -padding)
            top_right_brow_pts = face_kpts[range(46, 41, -1)] + (0, -padding)
            hole_mask_pts = np.concatenate([chin_pts, top_right_brow_pts, top_left_brow_pts], axis=0).astype(np.int32)

            cv2.fillPoly(hole_mask, [hole_mask_pts], (255, 255, 255))

        # blur
        kw = 10 * max(1, h / 512)
        hole_mask = cv2.blur(hole_mask, (kw, kw), borderType=cv2.BORDER_CONSTANT)

        # consider cases of hair covering these holes
        hole_mask = ((1.0 - self.soft_hair_mask / 255.0) * hole_mask).astype(np.uint8)

        return np.max([self.soft_skin_mask, hole_mask], axis=0)

    def get_hair_mask(self):
        assert self.sem_mask is not None, 'sem_mask is not available now'
        return pick_mask_by_color(self.sem_mask, self.mask_colors['hair'])

    def get_skin_mask(self):
        assert self.sem_mask is not None, 'sem_mask is not available now'
        return pick_mask_by_color(self.sem_mask, self.mask_colors['shin'])

    def get_cloth_mask(self):
        assert self.sem_mask is not None, 'sem_mask is not available now'
        return pick_mask_by_color(self.sem_mask, self.mask_colors['cloth'])

    def get_face_mask(self):
        assert self.sem_mask is not None, 'sem_mask is not available now'
        return pick_mask_by_color(self.sem_mask, self.mask_colors['face'])

    def get_portrait_mask(self):
        assert self.sem_mask is not None, 'sem_mask is not available now'
        bg_mask = pick_mask_by_color(self.sem_mask, self.mask_colors['bg'])
        portrait_mask = np.bitwise_not(bg_mask)
        return portrait_mask

    def get_super_mask(self, img_mask=None, test_mode=False):
        sem_mask = self.get_sem_mask(img_mask=img_mask)

        if sem_mask is None:
            return (None, None) if test_mode else None

        super_mask = self._add_fc_mask(sem_mask, self.multi_face_key_points)
        self.super_mask = super_mask

        return (sem_mask, super_mask) if test_mode else super_mask

    def _add_fc_mask(self, mask, multi_face_kpts):
        # TODO(ruanshihai): pts dimension is (n, 2) or (n, 1, 2)
        w, h = mask.shape[1], mask.shape[0]

        contours = []
        face_list = []
        for kpts in multi_face_kpts:
            for k, v in self.info_kpts_range.items()[1:]:
                pts = kpts[v[0]:v[1] + 1].copy()

                if k == 'left_brow' or k == 'right_brow':
                    # if self.is_A:
                    #     pts = self._morph_points(pts, (w, h), ksize=(3, 3), iterations=-3)
                    contours.append((pts, self.mask_colors['brow']))

                elif k == 'left_eye' or k == 'right_eye':
                    pts = cv2.convexHull(pts.astype(np.int32)[:, np.newaxis, :])
                    # if self.is_A:
                    #     pts = self._morph_points(pts, (w, h), ksize=(3, 3), iterations=5)
                    contours.append((pts, self.mask_colors['eye']))

                elif k == 'nose':
                    pts = cv2.convexHull(pts.astype(np.int32)[:, np.newaxis, :])
                    # if self.is_A:
                    #     pts = self._morph_points(pts, (w, h), ksize=(3, 1), iterations=-7)
                    contours.append((pts, self.mask_colors['nose']))

                elif k == 'mouth':
                    outer_pts, inner_pts = pts[0:12], pts[12:20]
                    # if self.is_A:
                    #     outer_pts = self._morph_points(outer_pts, (w, h), ksize=(3, 1), iterations=-3)
                    contours.append((outer_pts, self.mask_colors['mouth']))
                    contours.append((inner_pts, self.mask_colors['tooth']))

            lbrow = 1.5 * (kpts[33:37][::-1] - kpts[33:42].mean(0)) + kpts[33:42].mean(0)
            rbrow = 1.5 * (kpts[42:48][::-1] - kpts[42:51].mean(0)) + kpts[42:51].mean(0)

            face_list.append(
                np.concatenate(
                    (kpts[:32],kpts[42:46][::-1],kpts[33:37][::-1])
                ).astype(np.int32)
            )
        fc_mask = np.zeros(mask.shape,dtype=np.uint8)

        cv2.fillPoly(fc_mask,
                     face_list,
                     (0, 255, 255),
                     lineType=cv2.CV_AA)
        for contour, color in contours:
            cv2.fillPoly(fc_mask, [contour.astype(np.int32)], color, lineType=cv2.CV_AA)

        fc_mask[np.bitwise_and(mask[:,:,2]>10,mask[:,:,0]>10)] = 0
        other = fc_mask.mean(2)<10
        fc_mask[other] = mask[other]
        return fc_mask

    @staticmethod
    def _morph_points(pts, size, ksize=(3, 3), iterations=0):
        if iterations == 0:
            return pts

        mask = np.zeros(size, dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        if iterations < 0:
            mask = cv2.erode(mask, kernel, iterations=-iterations)
        else:
            mask = cv2.dilate(mask, kernel, iterations=iterations)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            return contour[::4].copy()
        else:
            return np.int32([]).reshape(0, 2)

    def get_sem_mask(self, img_mask=None):
        # result must be stored in self.sem_mask for subsequent usage
        if img_mask is None:
            self.sem_mask = self._get_sem_mask()
        else:
            self.sem_mask = self._get_ground_truth_sem_mask(img_mask)

        return self.sem_mask

    def _get_ground_truth_sem_mask(self, img_mask):
        sem_mask = self._recolor_ground_truth_mask(img_mask)
        return sem_mask

    def _recolor_ground_truth_mask(self, img_mask):
        # labeling standard of cartoon segmentation
        color_mapping_table = {
            (0, 0, 0): self.mask_colors['bg'],            # bg
            (255, 0, 255): self.mask_colors['bg'],        # 3rd kind

            # [1] hair region
            (194, 158, 241): self.mask_colors['hair'],    # hair

            # [2] cloth region
            (244, 206, 126): self.mask_colors['cloth'],   # cloth

            # [3] skin region
            (0, 100, 0): self.mask_colors['skin'],        # ears
            (51, 108, 153): self.mask_colors['skin'],     # neck
            (153, 247, 255): self.mask_colors['skin'],    # hand

            # [4] face region
            (0, 0, 255): self.mask_colors['face'],        # hairline
            (105, 194, 128): self.mask_colors['face'],    # extended hairline
            (169, 202, 204): self.mask_colors['face'],    # face
        }

        edited_mask = np.zeros(img_mask.shape, dtype=np.uint8)
        edited_mask[:] = self.mask_colors['others']
        for old_color, new_color in color_mapping_table.items():
            region = pick_mask_by_color(img_mask, old_color, thresh=20)
            edited_mask = np.bitwise_or(np.bitwise_and(edited_mask, np.bitwise_not(region)),
                                        np.bitwise_and(np.uint8(new_color), region))

        return edited_mask

    def _get_sem_mask(self):
        # TODO(ruanshihai): speed up calculation for images of large resolution
        soft_mask = self.get_soft_sem_mask()
        return soft_mask


class WarpedFaceSegment(FaceSegment):
    def __init__(self, resolution=(960, 1280), crop_size='XXL'):
        super(WarpedFaceSegment, self).__init__(resolution, crop_size)
        self.name = 'WarpedFaceSegment'

    def get_face_key_points(self, fa_points=None):
        afExtentParam = self.afExtentParams[self.crop_size]
        if fa_points is None:
            face_list = self.get_face_list(self.ori_img,
                                           afExtentParam,
                                           face_size=self.resolution,
                                           border_fill=self.border_value,
                                           warp_flag=0)
        else:
            face_list = self.get_face_list_by_fa_points(self.ori_img,
                                                        afExtentParam,
                                                        fa_points.reshape(-1),
                                                        face_size=self.resolution,
                                                        border_fill=self.border_value,
                                                        warp_flag=0)
        self.reserved_fdfa_face_list = face_list

        if len(face_list) == 0:
            return None, None

        face_list = face_list[0]
        self.face_list = face_list

        num_key_points = len(face_list[1]) / 4
        self.key_points = face_list[1][2*num_key_points:].reshape(-1, 2)
        self.face = face_list[0][:, :, 0:3].copy()
        self.valid_mask = np.repeat(face_list[0][:, :, 3:4], 3, axis=2)
        self.valid_rect = face_list[2][26:30].astype(np.int)

        # adjust multi-face key points to align the selected face
        warp_mat = np.array(self.face_list[2][18:24]).reshape(2, 3).transpose()
        self.multi_face_key_points = []
        for fl in self.reserved_fdfa_face_list:
            ori_kpts = fl[1][0:2*num_key_points].reshape(-1, 2)
            warp_kpts = np.matmul(np.concatenate([ori_kpts, np.ones((num_key_points, 1))], axis=1), warp_mat)
            self.multi_face_key_points.append(warp_kpts)

        return self.face, self.key_points

    def _get_ground_truth_sem_mask(self, img_mask):
        recolored_mask = self._recolor_ground_truth_mask(img_mask)
        sem_mask = self.ref_crop_another_img(recolored_mask, border_value=self.mask_colors['bg'])
        return sem_mask

    def _get_sy_sem_mask(self):
        # stage 1: extract valid bounding rectangle
        h0, w0 = self.ori_img.shape[0:2]
        rect_pts = self.face_list[2][8:16]
        x, y, w, h = cv2.boundingRect(rect_pts.reshape(4, 1, 2))
        vx, vy = max(0, x), max(0, y)
        vw, vh = min(w0, x+w)-vx, min(h0, y+h)-vy
        t, b, l, r = vy, vy+vh, vx, vx+vw
        valid_img = self.ori_img[t:b, l:r, :]

        # stage 2: resize to appropriate size
        ow, oh = int(self.face_list[2][16]), int(self.face_list[2][17])
        sw, sh = float(self.resolution[0])/ow, float(self.resolution[1])/oh
        nw, nh = int(vw*sw), int(vh*sh)
        valid_img = cv2.resize(valid_img, (nw, nh))

        # stage 3: adjust existing face key points
        num_key_points = len(self.face_list[1]) / 4
        ori_all_kpts = [fl[1][0:2*num_key_points].reshape(-1, 2) for fl in self.reserved_fdfa_face_list]
        adjusted_all_kpts = [(kpts-(vx, vy)) * (sw, sh) for kpts in ori_all_kpts]

        # stage 4: get refine mask
        face_seg = FaceSegment()
        face_seg.set_input(valid_img)
        face_seg.set_face_key_points(adjusted_all_kpts)
        valid_sem_mask = face_seg.get_sem_mask()

        # stage 5: inverse resize and construct a complete mask
        valid_sem_mask = cv2.resize(valid_sem_mask, (vw, vh))
        complete_sem_mask = np.zeros_like(self.ori_img, dtype=np.uint8)
        complete_sem_mask[t:b, l:r, :] = valid_sem_mask

        # stage 6: crop and warp
        sem_mask = self.ref_crop_another_img(complete_sem_mask, border_value=self.mask_colors['bg'])
        sem_mask = np.bitwise_and(sem_mask, self.valid_mask)

        return sem_mask

    def ref_crop_another_img(self, img, border_value=None):
        if border_value is None:
            border_value = self.border_value

        warp_mat = np.array(self.face_list[2][18:24]).reshape(2, 3)
        cropped_img = cv2.warpAffine(img, warp_mat, self.resolution, borderValue=border_value)

        return cropped_img


class NoWarpedFaceSegment(FaceSegment):
    def __init__(self, resolution=(960, 1280), crop_size='XXL'):
        super(NoWarpedFaceSegment, self).__init__(resolution, crop_size)
        self.name = 'NoWarpedFaceSegment'

    def get_face_key_points(self, fa_points=None, crop_multi_face=False):
        afExtentParam = self.afExtentParams[self.crop_size]
        if fa_points is None:
            face_list = self.get_face_list(self.ori_img,
                                           afExtentParam,
                                           face_size=self.resolution,
                                           border_fill=self.border_value,
                                           warp_flag=2)
        else:
            face_list = self.get_face_list_by_fa_points(self.ori_img,
                                                        afExtentParam,
                                                        fa_points.reshape(-1),
                                                        face_size=self.resolution,
                                                        border_fill=self.border_value,
                                                        warp_flag=2)
        self.reserved_fdfa_face_list = face_list

        if len(face_list) == 0:
            return None, None

        if crop_multi_face:
            self._crop_multi_face()
        else:
            self._crop_single_face()

        return self.face, self.key_points

    def _crop_single_face(self):
        face_list = self.reserved_fdfa_face_list[0]

        num_key_points = len(face_list[1]) / 4
        self.key_points = face_list[1][2*num_key_points:].reshape(-1, 2)
        self.face = face_list[0][:, :, 0:3].copy()
        # self.valid_mask = np.repeat(face_list[0][:, :, 3:4], 3, axis=2)

        h0, w0 = self.ori_img.shape[0:2]
        ox, oy, ow, oh = face_list[2][0:4].astype(np.int)
        ovx, ovy = max(0, ox), max(0, oy)
        ovw, ovh = min(w0, ox+ow)-ovx, min(h0, oy+oh)-ovy
        self.crop_rect = ox, oy, ow, oh
        self.crop_valid_rect = ovx, ovy, ovw, ovh
        self.crop_new_size = self.resolution
        self.valid_rect = face_list[2][26:30].astype(np.int)

        self.multi_face_key_points = []
        sw, sh = float(self.resolution[0]) / ow, float(self.resolution[1]) / oh
        for fl in self.reserved_fdfa_face_list:
            ori_kpts = fl[1][0:2 * num_key_points].reshape(-1, 2)
            adjusted_kpts = (ori_kpts - (ox, oy)) * (sw, sh)
            self.multi_face_key_points.append(adjusted_kpts)

    def _crop_multi_face(self):
        # TODO(ruanshihai): filter out some faces
        face_list = self.reserved_fdfa_face_list

        # find a bounding box containing all the cropped faces and find the biggest face
        faces_corner_pts = []
        faces_size = []
        for fl in face_list:
            x, y, w, h = fl[2][0:4].astype(np.int)
            faces_corner_pts.append((x, y))
            faces_corner_pts.append((x+w, y+h))
            faces_size.append((w, h))
        tl = np.min(faces_corner_pts, axis=0)
        br = np.max(faces_corner_pts, axis=0)
        max_idx = np.argmax([w*h for w, h in faces_size])
        max_w, max_h = faces_size[max_idx]

        h0, w0 = self.ori_img.shape[0:2]
        ox, oy, ow, oh = tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]
        ovx, ovy = max(0, ox), max(0, oy)
        ovw, ovh = min(w0, ox+ow)-ovx, min(h0, oy+oh)-ovy

        sw, sh = float(self.resolution[0]) / max_w, float(self.resolution[1]) / max_h
        nw, nh = int(ow*sw), int(oh*sh)
        # TODO(ruanshihai): small offset or error?
        vx, vy, vw, vh = int((ovx-ox)*sw), int((ovy-oy)*sh), int(ovw*sw), int(ovh*sh)

        self.crop_rect = ox, oy, ow, oh
        self.crop_valid_rect = ovx, ovy, ovw, ovh
        self.crop_new_size = nw, nh
        self.valid_rect = vx, vy, vw, vh

        # crop multiple faces
        valid_face = cv2.resize(self.ori_img[ovy:ovy+ovh, ovx:ovx+ovw], (vw, vh))
        face = np.zeros((nh, nw, 3), dtype=np.uint8)
        face[:] = self.border_value
        face[vy:vy+vh, vx:vx+vw, :] = valid_face
        self.face = face

        # calculate all facial key points
        num_key_points = len(face_list[0][1]) / 4
        self.multi_face_key_points = []
        for fl in face_list:
            ori_kpts = fl[1][0:2 * num_key_points].reshape(-1, 2)
            adjusted_kpts = (ori_kpts - (ox, oy)) * (sw, sh)
            self.multi_face_key_points.append(adjusted_kpts)

        # key points of the biggest face (yet not used in other place)
        self.key_points = self.multi_face_key_points[max_idx]

    def _get_ground_truth_sem_mask(self, img_mask):
        recolored_mask = self._recolor_ground_truth_mask(img_mask)
        sem_mask = self.ref_crop_another_img(recolored_mask, border_value=self.mask_colors['bg'])
        return sem_mask

    def _get_sy_sem_mask(self):
        vx, vy, vw, vh = self.valid_rect

        valid_face = self.face[vy:vy+vh, vx:vx+vw, :]
        adjusted_all_kpts = [kpts - (vx, vy) for kpts in self.multi_face_key_points]

        face_seg = FaceSegment()
        face_seg.set_input(valid_face)
        face_seg.set_face_key_points(adjusted_all_kpts)
        valid_sem_mask = face_seg.get_sem_mask()

        sem_mask = np.zeros_like(self.face, dtype=np.uint8)
        sem_mask[vy:vy+vh, vx:vx+vw, :] = valid_sem_mask

        return sem_mask

    def ref_crop_another_img(self, img, border_value=None):
        if border_value is None:
            border_value = self.border_value

        vx, vy, vw, vh = self.valid_rect
        ovx, ovy, ovw, ovh = self.crop_valid_rect

        valid_img = cv2.resize(img[ovy:ovy+ovh, ovx:ovx+ovw, :], (vw, vh))

        full_img = np.zeros_like(self.face, dtype=np.uint8)
        full_img[:] = border_value
        full_img[vy:vy+vh, vx:vx+vw, :] = valid_img

        return full_img