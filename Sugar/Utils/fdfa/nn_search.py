# -*- coding: utf-8 -*-

import cv2
import numpy as np
from collections import OrderedDict


class NNSearch(object):
    def __init__(self, db_kpts, exceptions):
        self.db_kpts = db_kpts
        self.exceptions = exceptions

        self.info_kpts_range = OrderedDict([
            ('face', (0, 32)),
            ('left_brow', (33, 41)),
            ('right_brow', (42, 50)),
            ('left_eye', (51, 60)),
            ('right_eye', (61, 70)),
            ('nose', (71, 85)),
            ('mouth', (86, 105))
        ])

        # weights for different facial points while finding nearest neighbor
        self.fa_weight = {
            'left_eye': [2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 4.0, 4.0],
            'right_eye': [2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 4.0, 4.0],
            'nose': [2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0],
            'mouth': [2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 4.0, 4.0, 4.0, 2.0, 4.0, 4.0, 4.0]
        }

    def get_nn_index(self, kpts, key):
        best_idx, best_d = None, None

        if key == 'brow' or key == 'eye':
            left_key, right_key = 'left_'+key, 'right_'+key

            begin1, end1 = self.info_kpts_range[left_key]
            left_kpts = kpts[begin1:end1 + 1]
            begin2, end2 = self.info_kpts_range[right_key]
            right_kpts = kpts[begin2:end2 + 1]

            for idx, ref_kpts in enumerate(self.db_kpts):
                if key in self.exceptions and idx in self.exceptions[key]:
                    continue

                left_ref_kpts = ref_kpts[begin1:end1 + 1]
                right_ref_kpts = ref_kpts[begin2:end2 + 1]
                d = self.dist(left_kpts, left_ref_kpts, left_key) + self.dist(right_kpts, right_ref_kpts, right_key)
                if best_d is None or d < best_d:
                    best_idx, best_d = idx, d

        else:
            begin, end = self.info_kpts_range[key]
            kpts = kpts[begin:end + 1]

            for idx, ref_kpts in enumerate(self.db_kpts):
                if key in self.exceptions and idx in self.exceptions[key]:
                    continue

                ref_kpts = ref_kpts[begin:end + 1]
                d = self.dist(kpts, ref_kpts, key)
                if best_d is None or d < best_d:
                    best_idx, best_d = idx, d

        return best_idx

    def dist(self, kpts0, ref_kpts0, key):
        # normalize points
        x1, y1, w1, h1 = cv2.boundingRect(kpts0[:, np.newaxis, :].astype(np.float32))
        x2, y2, w2, h2 = cv2.boundingRect(ref_kpts0[:, np.newaxis, :].astype(np.float32))
        kpts = (kpts0 - (x1, y1)) / (w1, h1)
        ref_kpts = (ref_kpts0 - (x2, y2)) / (w2, h2)

        d = np.square(kpts - ref_kpts)
        if key in self.fa_weight:
            d = np.sum(d[:, 0]*self.fa_weight[key]) + np.sum(d[:, 1]*self.fa_weight[key])
        else:
            d = np.sum(d)

        # add constraints for special facial components
        if key == 'left_eye' or key == 'right_eye':
            # consider cases of closed eyes
            f1 = max(0.0, (kpts0[6][1]-kpts0[2][1])) / w1
            f2 = max(0.0, (ref_kpts0[6][1]-ref_kpts0[2][1])) / w2
            if (f1 < 0.20) != (f2 < 0.15):
                d += 1000.0

        elif key == 'mouth':
            d += 4 * np.sum(np.square(((kpts[19] - kpts[13]) - (ref_kpts[19] - ref_kpts[13]))))
            d += 4 * np.sum(np.square(((kpts[18] - kpts[14]) - (ref_kpts[18] - ref_kpts[14]))))
            d += 4 * np.sum(np.square(((kpts[17] - kpts[15]) - (ref_kpts[17] - ref_kpts[15]))))

            # evaluate the degree of mouth closure
            idxs = [[19, 18, 17], [13, 14, 15], [10, 9, 8], [2, 3, 4]]
            f1 = np.abs(np.mean(kpts[idxs[0]][1])-np.mean(kpts[idxs[1]][1])) \
                    / np.abs(np.mean(kpts[idxs[2]][1])-np.mean(kpts[idxs[3]][1]))
            f2 = np.abs(np.mean(ref_kpts[idxs[0]][1])-np.mean(ref_kpts[idxs[1]][1])) \
                    / np.abs(np.mean(ref_kpts[idxs[2]][1])-np.mean(ref_kpts[idxs[3]][1]))

            # # method 1: multi-level difference evaluation
            # d += int(np.abs(f1 - f2) / 0.1) * 1000.0

            # method 2: eyes both closed or both opened
            if (f1 < 0.15) != (f2 < 0.10):
                d += 1000.0

        return d
