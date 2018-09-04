# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import copy

import MT2Python
from fdfa.face_segment_server import FaceSegmentServer
from SkinRevisedMask import FaceInterface
from CVlib_for_Python.lib.Smooth_functional import MaskSmooth


def get_hard_optimizied_sem_mask(img, complexity=6, warp_choice=False, crop_size=None):
    om = OptimizeMask()
    om.set_input(img)
    mask = om.generate_hard_optimized_mask(complexity, False, warp_choice, crop_size)
    return mask


class OptimizeMask(object):
    def __init__(self):
        self.resolution = 1280
        self.crop_size = 'XXL'
        self.afExtentParams = {
            'M': [0.6, 0.0, 0.0, 0.0, 0.3],
            'L': [1.0, 0.0, 0.6, 0.0, 0.0],
            'XL': [1.0, 0.0, 1.2, 0.0, 0.0],
            'XXL': [1.1, 0.0, 1.6, 0.0, 0.0],
        }

    def set_input(self, img):
        self.ori_img = img
        self.img = img

    def generate_hard_optimized_mask(self, complexity, using_optimized_face_mask, warp_choice, crop_size):
        if crop_size is None:
            crop_size = self.crop_size

        # init here to get the keypoints
        face_reserve_judgement = self.get_face_warp_points(warp_choice,  crop_size)

        if not face_reserve_judgement:
            return None

        # get skin/ hair / portrait mask by server
        server = FaceSegmentServer(self.img)

        # notice the skin_mask here @TODO
        # skin_mask = server.getSkin(self.img)
        portrait_mask = server.getPortrait(self.img)

        # get hair mask here
        self.soft_hair_mask = server.getHair(self.img)
        self.soft_hair_mask_src = copy.deepcopy(self.soft_hair_mask)
        self.soft_hair_mask = cv2.resize(self.soft_hair_mask, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_CUBIC)

        #  get skin_mask here
        self.soft_skin_mask = server.getSkin(self.img)
        self.soft_skin_mask_src = copy.deepcopy(self.soft_skin_mask)

        self.soft_skin_mask = cv2.resize(self.soft_skin_mask, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_CUBIC)
        self.soft_skin_mask = np.bitwise_not(self.soft_skin_mask)

        # fill the holes of face
        self.soft_skin_mask = self.fill_skin_mask_holes(self.soft_skin_mask, self.soft_hair_mask, self.key_points,  self.img.shape[0])
        # self.soft_skin_mask_backup = self.__fill_facial_component_holes()


        if using_optimized_face_mask:
            face_part_mask = server.getWholeFacePart(img)
            # get foreground
            face_foreground = cv2.bitwise_not(server.getFacePart(face_part_mask, 'background', 512))
            # get ear
            left_ear_server_foreground = server.getFacePart(face_part_mask, 'left_ear', 512)
            right_ear_server_foreground = server.getFacePart(face_part_mask, 'right_ear', 512)
            left_ear_server_foreground = self.morph_sequence(left_ear_server_foreground, 3, 'close', 1)
            right_ear_server_foreground = self.morph_sequence(right_ear_server_foreground, 3, 'close', 1)
            # get fore face
            face_mask = face_foreground - left_ear_server_foreground - right_ear_server_foreground

        else:
            # face mask complexity
            faceOptimized = FaceInterface.FaceOptimized()
            faceOptimized.set_input(self.img)
            faceOptimized.set_leastsq_complexity(complexity)
            # get face mask
            # faceOptimized.set_face_list(self.face_list)
            face_mask = faceOptimized.get_face_mask()
        if len(face_mask) > 0:
            # in order to get the fill component skin
            # face_mask_poly = cv2.resize(self.get_face_mask_poly(thresh=140), (512, 512), interpolation=cv2.INTER_CUBIC)
            # face_mask = cv2.resize(face_mask, (512, 512), interpolation=cv2.INTER_CUBIC)
            # # combine th poly mask with src face mask
            # # @todo using the kx+b to combine the mask here 9.29
            # face_mask_poly += face_mask - cv2.bitwise_and(face_mask_poly, face_mask)
            # face_mask = face_mask_poly

            # @todo

            h = face_mask.shape[0]
            w = face_mask.shape[1]
            face_mask_map = self.get_line_as_limitation_map(self.key_points, h, w, 'above', 8, 24)

            # IMPORTANT PARAMETER HERE @YANGMING
            face_mask[face_mask_map] = (0, 0, 0)

            # src mask with the fa face
            face_mask_poly = cv2.resize(self.get_face_mask_poly(8, 24), (h, w), interpolation=cv2.INTER_CUBIC)
            face_mask = cv2.bitwise_or(face_mask_poly, face_mask)
            self.face_mask = face_mask

            # resize the soft_skin as restriction
            self.soft_skin_mask = cv2.resize(self.soft_skin_mask,  (512, 512), interpolation=cv2.INTER_CUBIC)
            skin_mask = self.soft_skin_mask

            # resize the hard_skin as restriction
            self.soft_hair_mask = cv2.resize(self.soft_hair_mask,  (512, 512), interpolation=cv2.INTER_CUBIC)
            hair_mask = self.soft_hair_mask

            total_mask = [skin_mask, hair_mask, portrait_mask, face_mask]
            # resize mask and threshold
            threshold_list = [100, 170, 100, 100]
            total_mask = [cv2.threshold(total_mask[i], threshold_list[i], 255, cv2.THRESH_BINARY)[1] for i in
                          range(0, 4)]
            skin_mask, hair_mask, portrait_mask, face_mask = [
                cv2.resize(total_mask[i], (512, 512), interpolation=cv2.INTER_CUBIC) for i in range(0, 4)]

            self.skin_mask = skin_mask
            self.hair_mask = hair_mask
            self.potrait_mask = portrait_mask
            self.face_mask = face_mask

            # final output whole mask
            default_color = (0, 0, 0)
            mask = np.zeros(face_mask.shape, dtype=np.uint8)
            mask[:] = default_color

            # confident face mask with skin judgement
            # @todo notice the processing sequence here! put the soft_skin_mask firstly
            confident_face_mask_region = cv2.bitwise_and(self.skin_mask, face_mask)
            # fill region
            confident_face_mask_region = cv2.bitwise_not(FaceInterface.FaceOptimized.getMaxRegion(cv2.bitwise_not(confident_face_mask_region)))

            # @todo here is the judgement of the cut final region
            confident_cut_face_region = face_mask - confident_face_mask_region
            # cv2.imshow('confident_cut_face_region_as_hair', confident_cut_face_region)
            # confident hair mask
            confident_hair_mask_region = cv2.bitwise_and(cv2.bitwise_not(hair_mask), cv2.bitwise_not(portrait_mask))
            # confident_hair_mask_region += confident_cut_face_region

            confident_hair_mask_region = self.morph_sequence(confident_hair_mask_region, 3, 'erode', 5)
            confident_hair_mask_region = self.morph_sequence(confident_hair_mask_region, 3, 'dilate', 5)
            confident_hair_mask_region = self.morph_sequence(confident_hair_mask_region, 5, 'open', 1)

            if using_optimized_face_mask:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
                confident_hair_mask_region = cv2.dilate(confident_hair_mask_region, kernel)
                confident_hair_mask_optimized_region = confident_hair_mask_region - cv2.bitwise_and(
                    confident_hair_mask_region,
                    confident_face_mask_region)
                confident_hair_mask_optimized_region = cv2.bitwise_and(confident_hair_mask_optimized_region, hair_mask)

            # confident skin mask
            confident_skin_mask_region = cv2.bitwise_and(cv2.bitwise_not(confident_face_mask_region), self.skin_mask)
            confident_skin_mask_region = self.morph_sequence(confident_skin_mask_region, 3, 'erode', 5)
            confident_skin_mask_region = self.morph_sequence(confident_skin_mask_region, 3, 'dilate', 5)

            # condifident cloth mask
            head_mask = np.zeros(face_mask.shape, dtype=np.uint8)
            head_mask[:] = default_color
            head_mask[(np.all(confident_skin_mask_region > 0, axis=2))] = (255, 255, 255)
            head_mask[(np.all(confident_hair_mask_region > 0, axis=2))] = (255, 255, 255)
            head_mask[(np.all(confident_face_mask_region > 0, axis=2))] = (255, 255, 255)

            confident_head_mask_region = cv2.bitwise_and(cv2.bitwise_not(portrait_mask), head_mask)
            confident_cloth_mask_region = cv2.bitwise_not(portrait_mask) - confident_head_mask_region

            if using_optimized_face_mask:
                confident_cut_mask = cv2.bitwise_and(portrait_mask, head_mask)
                confident_hair_cut_mask_region = cv2.bitwise_and(confident_cut_mask, confident_hair_mask_region)
                confident_hair_cut_optimized_mask_region = confident_hair_cut_mask_region - cv2.bitwise_and(
                    confident_hair_cut_mask_region, confident_face_mask_region)

            # use nose as limitation
            mask_min_y = self.GetRegionMinArea(self.key_points, begin=76, end=84)

            # add judgement for different size
            mask_min_y = mask_min_y * (512.0 / self.img.shape[0])
            confident_cloth_mask_region[0:int(mask_min_y), :, :] = 0

            confident_cloth_mask_region = self.morph_sequence(confident_cloth_mask_region, 3, 'erode', 5)
            confident_cloth_mask_region = self.morph_sequence(confident_cloth_mask_region, 3, 'dilate', 5)
            # hair = cv2.bitwise_and(confident_hair_cut_optimized_mask_region, confident_hair_mask_region)

            skin_mask_color = (255, 255, 0)
            hair_mask_color = (255, 0, 255)
            cloth_mask_color = (255, 0, 0)
            face_mask_color = (0, 255, 255)

            # replace confident hair mask
            face_hair_mask = cv2.bitwise_not(cv2.bitwise_or(face_mask, confident_hair_mask_region))
            confident_hair_mask_region = FaceInterface.FaceOptimized.filterRegion(cv2.bitwise_not(face_hair_mask), 200)
            # adding the bg as limitation
            confident_cloth_mask_region -= cv2.bitwise_and(confident_cloth_mask_region, portrait_mask)
            confident_hair_mask_region -= cv2.bitwise_and(confident_hair_mask_region, portrait_mask)
            confident_skin_mask_region -= cv2.bitwise_and(confident_skin_mask_region, portrait_mask)
            confident_face_mask_region -= cv2.bitwise_and(confident_face_mask_region, portrait_mask)

            confident_cloth_mask_region = self.optimzied_smooth_mask(confident_cloth_mask_region, 5)
            confident_hair_mask_region = self.optimzied_smooth_mask(confident_hair_mask_region, 10)
            confident_skin_mask_region = self.optimzied_smooth_mask(confident_skin_mask_region, 5)
            confident_face_mask_region = self.optimzied_smooth_mask(confident_face_mask_region, 8)

            # fillting the small region of mask
            confident_skin_mask_region = FaceInterface.FaceOptimized.filterRegion(confident_skin_mask_region)
            confident_cloth_mask_region = FaceInterface.FaceOptimized.filterRegion(confident_cloth_mask_region)

            if using_optimized_face_mask:
                mask[(np.all(confident_skin_mask_region > 0, axis=2))] = skin_mask_color
                mask[(np.all(confident_hair_mask_region > 0, axis=2))] = hair_mask_color
                mask[(np.all(confident_cloth_mask_region > 0, axis=2))] = cloth_mask_color
                mask[(np.all(confident_face_mask_region > 0, axis=2))] = face_mask_color
                mask[(np.all(left_ear_server_foreground > 0, axis=2))] = skin_mask_color
                mask[(np.all(right_ear_server_foreground > 0, axis=2))] = skin_mask_color

            else:
                mask[(np.all(confident_cloth_mask_region > 0, axis=2))] = (255, 255, 255)
                mask[(np.all(confident_hair_mask_region > 0, axis=2))] = (255, 255, 255)
                mask[(np.all(confident_skin_mask_region > 0, axis=2))] = (255, 255, 255)
                mask[(np.all(confident_face_mask_region > 0, axis=2))] = (255, 255, 255)
                cut_mask = mask - cv2.bitwise_not(FaceInterface.FaceOptimized.getMaxRegion(cv2.bitwise_not(mask)))

                masks_list = [confident_cloth_mask_region, confident_hair_mask_region,
                              confident_skin_mask_region, confident_face_mask_region]

                cloth_mask_hair = cv2.bitwise_and(confident_cloth_mask_region, confident_hair_mask_region)
                cloth_mask_skin = cv2.bitwise_and(confident_cloth_mask_region, confident_skin_mask_region)
                cloth_mask_face = cv2.bitwise_and(confident_cloth_mask_region, confident_face_mask_region)

                confident_cloth_mask_region = np.bitwise_and(confident_cloth_mask_region, np.bitwise_not(cloth_mask_hair))
                confident_cloth_mask_region = np.bitwise_and(confident_cloth_mask_region, np.bitwise_not(cloth_mask_skin))
                confident_cloth_mask_region = np.bitwise_and(confident_cloth_mask_region, np.bitwise_not(cloth_mask_face))


                hair_mask_skin = cv2.bitwise_and(confident_hair_mask_region, confident_skin_mask_region)
                hair_mask_face = cv2.bitwise_and(confident_hair_mask_region, confident_face_mask_region)

                confident_hair_mask_region = np.bitwise_and(confident_hair_mask_region, np.bitwise_not(hair_mask_skin))
                confident_hair_mask_region = np.bitwise_and(confident_hair_mask_region, np.bitwise_not(hair_mask_face))

                skin_mask_face = cv2.bitwise_and(confident_skin_mask_region, confident_face_mask_region)
                confident_skin_mask_region = np.bitwise_and(confident_skin_mask_region, np.bitwise_not(skin_mask_face))

                for i in range(0, 4):
                    masks_list[i] = np.bitwise_and(masks_list[i], np.bitwise_not(cut_mask))

                total_color = [cloth_mask_color, hair_mask_color, skin_mask_color , face_mask_color]
                masks_list = [cv2.resize(mask, (self.img.shape[1], self.img.shape[0]),interpolation=cv2.INTER_CUBIC) for mask in masks_list]
                sem_mask = np.zeros(self.img.shape, dtype=np.uint8)
                sem_mask[:] = default_color
                for i in range(0, 4):
                    sem_mask = np.bitwise_and(sem_mask, np.bitwise_not(masks_list[i])) + np.bitwise_and(
                        masks_list[i], np.uint8(total_color[i]))
                    self.smooth_edge_transition(sem_mask, masks_list[i], total_color[i])

                sem_mask = cv2.bitwise_and(sem_mask, cv2.merge([self.mask_limitation, self.mask_limitation, self.mask_limitation]))
                # post process
                # sem_mask_hair = sem_mask[np.all(mask == hair_mask_color)] = (255, 255, 255)
                # skin_min_y = self.GetRegionMinArea(self.key_points, begin=33, end=50)
                # skin_min_y = skin_min_y * (512.0 / self.img.shape[0])
                # output = self.GetHighLightCurve(self.img, masks_list[1])
                # @todo using lightCurve as limitation
                sem_mask = cv2.resize(sem_mask, (self.resolution, self.resolution))

            return sem_mask
        else:
            print "cannot find the face in this image, did you provide the right one?"
            return None

    def get_face_warp_points(self, warp_choice, crop_size):
        afExtentParam = self.afExtentParams[crop_size]
        face_list = MT2Python.FDFA.DetectWarpedFaceImages(self.img,
                                                              afExtentParam,
                                                              face_size=1280,
                                                              fd_score=0,
                                                              border_mode=MT2Python.MT_BORDER_CONSTANT,
                                                              border_fill=0)
        if not face_list:
            return []
        else:
            if warp_choice:
                self.img = face_list[0][0][:, :, 0:3].copy()
                self.mask_limitation = face_list[0][0][:, :, 3:4].copy()
                # set the warp begin index here
                warp_begin_index = 212
                self.key_points = face_list[0][1][warp_begin_index:].reshape(-1, 2)

            else:
                scale_face_list,_, self.mask_limitation = copy.deepcopy(self.no_align_crop_face(face_list[0], self.img, resolution=1280))
                self.img = copy.deepcopy(scale_face_list[0][:, :, 0:3])
                # set the warp begin index here
                warp_begin_index = 212
                self.key_points =scale_face_list[1][warp_begin_index:].reshape(-1, 2)

            return self.img, self.key_points

    def no_align_crop_face(self, each_face_list, img, resolution):

        scale_face_list = list(each_face_list)
        w, h = int(scale_face_list[2][16]), int(scale_face_list[2][17])
        rect_pts = scale_face_list[2][8:16]
        cx, cy = int(sum(rect_pts[0::2]) / 4), int(sum(rect_pts[1::2]) / 4)

        # get orientation distance
        x, y = cx - w / 2, cy - h / 2
        face = np.zeros((h, w, 3), dtype=np.uint8)
        face[:] = (0, 0, 0)

        # new points ax ay
        ax, ay = max(0, -x), max(0, -y)
        # origin points
        bx, by = max(0, x), max(0, y)
        nw, nh = min(w - ax, img.shape[1] - bx), min(h - ay, img.shape[0] - by)
        input_image = img[by:by + nh, bx:bx + nw]
        face[ay:ay + nh, ax:ax + nw] = input_image

        mask_limitation = np.zeros((h, w, 3), dtype=np.uint8)
        mask_limitation[:] = (0, 0, 0)
        mask_limitation[ay:ay + nh, ax:ax + nw] = 255

        # rescale face to desired resolution
        scale_face_list[0] = cv2.resize(face, (resolution, resolution),
                                        interpolation=cv2.INTER_AREA)

        # fix facial key points
        scale_w, scale_h = float(resolution) / w, float(resolution) / h
        num_key_points = len(scale_face_list[1]) / 4
        for i in range(num_key_points):
            scale_face_list[1][2 * num_key_points + 2 * i] = (scale_face_list[1][2 * i] - x) * scale_w
            scale_face_list[1][2 * num_key_points + 2 * i + 1] = (scale_face_list[1][2 * i + 1] - y) * scale_h
        mask_limitation = cv2.resize(mask_limitation, (resolution, resolution),
                                        interpolation=cv2.INTER_AREA)
        return scale_face_list, input_image, mask_limitation[:,:,0]

    @staticmethod
    def fill_skin_mask_holes(skin_mask, hair_mask, keypoints, shape_h):
        # detect face points
        chin_pts = keypoints[range(0, 33, 1)]
        padding = 8 * max(1, shape_h / 512)
        top_left_brow_pts = keypoints[range(37, 32, -1)] + (0, -padding)
        top_right_brow_pts = keypoints[range(46, 41, -1)] + (0, -padding)
        # get the outer face points
        hole_mask_pts = np.concatenate([chin_pts, top_right_brow_pts, top_left_brow_pts], axis=0).astype(np.int32)
        hole_mask = np.zeros(skin_mask.shape, dtype=np.uint8)
        cv2.fillPoly(hole_mask, [hole_mask_pts], (255, 255, 255))
        # cut the hair mask region here as restriction when we meet the situation that hair mask covered the skin
        hole_mask = (hair_mask / 255.0 * hole_mask).astype(np.uint8)
        # blur mask
        kw = 10 * max(1, skin_mask.shape[0] / 512)
        hole_mask = cv2.blur(hole_mask, (kw, kw), borderType=cv2.BORDER_CONSTANT)
        # get the max value of two images
        skin_mask = np.max([skin_mask, hole_mask], axis=0)
        return skin_mask

    def get_face_mask_poly(self, begin=10, end=21):
        # if not self.mask_available:
        #     self.__generate_masks()

        chin = self.key_points[range(33)[begin:end]].astype(np.int32)
        # chin[0][1], chin[-1][1] = 0, 0
        mask = np.zeros(self.img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [chin], (255, 255, 255))
        return mask

    @staticmethod
    def get_face_optimized_mask(img, complexity):
        # face mask complexity
        faceOptimized = FaceInterface.FaceOptimized()
        faceOptimized.set_input(img)
        faceOptimized.set_leastsq_complexity(complexity)

        # get face mask
        face_mask = faceOptimized.get_face_mask()
        return face_mask

    def get_line_as_limitation_map(self, key_points, h, w, orientation, begin=10, end=21):
        outer_face_line = key_points[range(33)[begin:end]]

        x1 = outer_face_line[-1][0]
        x2 = outer_face_line[0][0]
        y1 = outer_face_line[-1][1]
        y2 = outer_face_line[0][1]

        k = (y2 - y1) / (x2 - x1)
        b = y2 - x2 * k
        y_map = np.array([range(h)]).transpose(1, 0).repeat(w, 1)
        x_map = np.array([range(w)]).repeat(h, 0)
        if orientation == 'above':
            position_map = y_map > x_map * k + b
        else:
            position_map = y_map < x_map * k + b
        return position_map


    # get the smooth mask contour
    def smooth_mask(self, mask, color, kernel_A, kernel_B):
        default_color = (0, 0, 0)
        smooth_mask = np.zeros(mask.shape, dtype=np.uint8)
        smooth_mask[:] = default_color
        smooth_mask[(np.all(mask == color, axis=2))] = color
        # smooth_mask = self.morph_sequence(smooth_mask, 5, 'close', 1)
        mask = self.smooth_edge_transition(mask, smooth_mask, kernel_A=kernel_A, kernel_B=kernel_B, color=color)
        return mask

    # get the smooth mask contour : method 2
    @staticmethod
    def optimzied_smooth_mask(mask, kernel_size, color=[]):
        default_color = (0, 0, 0)
        smooth_mask = np.zeros(mask.shape, dtype=np.uint8)
        smooth_mask[:] = default_color

        # smooth_mask[(np.all(mask == color, axis=2))] = (255,255,255)
        smooth_mask[(np.all(mask > 0, axis=2))] = (255, 255, 255)

        smooth_mask = MaskSmooth(smooth_mask[:, :, 1], kernel_size)
        smooth_mask = np.uint8(cv2.merge([smooth_mask, smooth_mask, smooth_mask]))
        if color:
            smooth_mask[(np.all(smooth_mask > 0, axis=2))] = color
        return smooth_mask

    @staticmethod
    def smooth_edge_transition(img, mask, color, kernel_A = 15, kernel_B=10):
        # former paramter is max interval of sampling
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(mask[:, :, 0].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        reserved = []
        min_area = 100.0 / (512 * 512) * (img.shape[0] * img.shape[1])
        for elem in contours:
            if cv2.contourArea(elem) > min_area:
                # sample less points to make edge smoother
                reserved.append(elem[::min(kernel_A, elem.shape[0] / kernel_B)].copy())

        cv2.drawContours(img, reserved, -1, color, 3, lineType=cv2.CV_AA)
        return img

    @staticmethod
    def draw_fa_points(img, key_points, begin=86, end=106):
        # draw the fa to check
        for p in key_points[begin:end]:
            cv2.circle(img, (p[0], p[1]), 2, (0, 0, 255), -1)
        return img

    @staticmethod
    def morph_sequence(mask, kernel_size, morph_choice, iret):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if morph_choice == "erode":
            mask = [cv2.erode(mask, kernel) for i in range(0, iret)][-1]
        if morph_choice == "dilate":
            mask = [cv2.dilate(mask, kernel) for i in range(0, iret)][-1]
        if morph_choice == "open":
            mask = [cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) for i in range(0, iret)][-1]
        if morph_choice == "close":
            mask = [cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) for i in range(0, iret)][-1]

        return mask

    @staticmethod
    def GetRegionMinArea(keypoints, begin=86, end=105):
        # default mouth
        y_value_min = keypoints[begin][1]
        for y in range(begin, end + 1):
            # From 51 to 61
            if y_value_min > keypoints[y][1]:
                y_value_min = keypoints[y][1]
        return y_value_min

    @staticmethod
    def GetFaceDire(img_src):
        afExtentParam = [1.0, 0.0, 0.6, 0.0, 0.0]
        face_list = MT2Python.FDFA.DetectWarpedFaceImages(img_src,
                                                          afExtentParam,
                                                          face_size=-1,
                                                          fd_score=0,
                                                          border_mode=MT2Python.MT_BORDER_CONSTANT,
                                                          border_fill=(0, 0, 0))

        return face_list[0][3],face_list[0][1]

    @staticmethod
    def Get3DMat(angle):

        x_a,y_a,z_a = angle*3.1415926/180*(-1,-1,-1)
        mat_x = [
            [1,0,0],
            [0,np.cos(x_a),-np.sin(x_a)],
            [0,np.sin(x_a),np.cos(x_a)]]

        mat_y = [
            [np.cos(y_a),0, -np.sin(y_a)],
            [0, 1, 0],
            [np.sin(y_a), 0,np.cos(y_a)]]

        mat_z = [
            [np.cos(z_a), -np.sin(z_a),0],
            [np.sin(z_a), np.cos(z_a),0],
            [0, 0, 1]

        ]
        return np.array([mat_x,mat_y,mat_z])

    @staticmethod
    def Use3DMat(vector,mat_list):
        xyz_ = np.matmul(
            np.matmul(
                np.matmul(
                    vector, mat_list[0]
                )
                , mat_list[1]
            ),
            mat_list[2]
        )
        return xyz_

    def GetHighLightCurve(self, src, hair_mask):
        point, face_point = self.GetFaceDire(src)
        half_w = min(src.shape[0] ,src.shape[1])// 2

        a_x = int(face_point[73 * 2])
        a_y = int(face_point[73 * 2 + 1])
        b_x = int(face_point[16 * 2])
        b_y = int(face_point[16 * 2 + 1])

        c_x = int(face_point[4 * 2])
        c_y = int(face_point[4 * 2 + 1])
        d_x = int(face_point[28 * 2])
        d_y = int(face_point[28 * 2 + 1])
        len_ab = ((a_x - b_x) ** 2 + (a_y - b_y) ** 2) ** 0.5
        len_cd = ((c_x - d_x) ** 2 + (c_y - d_y) ** 2) ** 0.5
        mat_xyz = self.Get3DMat(point[:3])
        center = self.Use3DMat(np.array([[-1, -len_ab, 0], [1, -len_ab, 0]]), mat_xyz)
        dire_center = center[1] - center[0]
        dire_center /= (dire_center ** 2).sum() ** 0.5
        mask_max_v = 0
        for i in range(half_w):
            x = np.clip(half_w + center[0][0] + dire_center[0] * i, 0, half_w * 2 - 2)
            y = np.clip(half_w + center[0][1] + dire_center[1] * i, 0, half_w * 2 - 2)

            if hair_mask[int(y), int(x), 1] > 0:
                mask_max_v = max(i, mask_max_v)
            x = np.clip(half_w + center[0][0] - dire_center[0] * i, 0, half_w * 2 - 2)
            y = np.clip(half_w + center[0][1] - dire_center[1] * i, 0, half_w * 2 - 2)
            if hair_mask[int(y), int(x), 1] > 0:
                mask_max_v = max(i, mask_max_v)
        len_cd = mask_max_v
        r = np.array(range(360)) * 3.1415926 / 180
        x = np.sin(r) * len_cd
        z = np.cos(r) * len_cd
        y = x.copy()
        y[:] = -len_ab * 1.3
        xyz = np.array([x, y, z]).transpose(1, 0)

        xyz_ = self.Use3DMat(xyz, mat_xyz)
        # xyz_ = xyz
        xyz_[:, 2] = xyz_[:, 2] / xyz_[:, 2].max()
        x = xyz_[:, 0]  # * np.exp(-xyz_[:, 2])
        y = xyz_[:, 1]  # * np.exp(-xyz_[:, 2])
        x = x.astype(int)
        y = y.astype(int)
        z = xyz_[:, 2]

        x_center = int(point[3] + half_w)
        y_center = int(point[4] + half_w)
        highlight = []
        for i in range(x.shape[0]):
            if z[i]<0:
                src[y[i] + y_center:y[i] + y_center+2,
                x[i] + x_center:x[i] + x_center+2, :] = (0, 0, z[i]*255)
                highlight.append([x[i] + x_center,y[i] + y_center])
        return np.array(highlight, dtype=float)
