#!/usr/bin/env python

import numpy as np
from math import cos, sin, pi
import cv2
import random

from config import cfg
from skeletons.skeleton_config_base import SkeletonConfigBase


class AugmentSelection:
    def __init__(self, flip=False, degree=0., crop=(0, 0), scale=1.):
        self.flip = flip
        self.degree = degree  # rotate
        self.crop = crop  # shift actually
        self.scale = scale

    @staticmethod
    def random():
        flip = random.uniform(0., 1.) > cfg.train.augmentation.flip_prob
        degree = random.uniform(-1., 1.) * cfg.train.augmentation.max_rotate_degree
        scale = 1
        if random.uniform(0., 1.) > cfg.train.augmentation.scale_prob:
            # TODO: see 'scale improbability' TODO above
            scale = (cfg.train.augmentation.scale_max - cfg.train.augmentation.scale_min) * random.uniform(0., 1.) + \
                    cfg.train.augmentation.scale_min
        x_offset = int(random.uniform(-1., 1.) * cfg.train.augmentation.center_perterb_max)
        y_offset = int(random.uniform(-1., 1.) * cfg.train.augmentation.center_perterb_max)

        return AugmentSelection(flip, degree, (x_offset, y_offset), scale)

    @staticmethod
    def unrandom():
        flip = False
        degree = 0.
        scale = 1.
        x_offset = 0
        y_offset = 0

        return AugmentSelection(flip, degree, (x_offset, y_offset), scale)

    def affine(self, center, scale_self):
        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards
        A = self.scale * cos(self.degree / 180. * pi)
        B = self.scale * sin(self.degree / 180. * pi)

        divisor = scale_self * self.scale
        if divisor <= 0:
            divisor = 1 # prevent zero division

        scale_size = cfg.train.augmentation.target_dist / divisor

        (width, height) = center
        center_x = width + self.crop[0]
        center_y = height + self.crop[1]

        center2zero = np.array([[1., 0., -center_x],
                                [0., 1., -center_y],
                                [0., 0., 1.]])

        rotate = np.array([[A, B, 0],
                           [-B, A, 0],
                           [0, 0, 1.]])

        scale = np.array([[scale_size, 0, 0],
                          [0, scale_size, 0],
                          [0, 0, 1.]])

        flip = np.array([[-1 if self.flip else 1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])

        center2center = np.array([[1., 0., cfg.general.input_width // 2],
                                  [0., 1., cfg.general.input_height // 2],
                                  [0., 0., 1.]])

        # order of combination is reversed
        combined = center2center.dot(flip).dot(scale).dot(rotate).dot(center2zero)

        return combined[0:2]


def transform(img, mask, meta, mask_shape: (int, int), skeleton_config: SkeletonConfigBase, aug: AugmentSelection = AugmentSelection.unrandom()):
    # warp picture and mask
    M = aug.affine(meta['objpos'][0], meta['scale_provided'][0])

    img = cv2.warpAffine(img, M, (cfg.general.input_height, cfg.general.input_width), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
    mask = cv2.warpAffine(mask, M, (cfg.general.input_height, cfg.general.input_width), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    mask = cv2.resize(mask, mask_shape, interpolation=cv2.INTER_CUBIC)
    # _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    # assert np.all((mask == 0) | (mask == 255)), "Interpolation of mask should be thresholded only 0 or 255\n" + str(mask)
    mask = mask.astype(np.float) / 255.

    original_points = meta['joints'].copy()
    original_points[:, :, 2] = 1  # we reuse 3rd column in completely different way here, it is hack
    converted_points = np.matmul(M, original_points.transpose([0, 2, 1])).transpose([0, 2, 1])
    meta['joints'][:, :, 0:2] = converted_points

    # we just made image flip, i.e. right leg just became left leg, and vice versa

    if aug.flip:
        tmpLeft = meta['joints'][:, skeleton_config.left_parts, :]
        tmpRight = meta['joints'][:, skeleton_config.right_parts, :]
        meta['joints'][:, skeleton_config.left_parts, :] = tmpRight
        meta['joints'][:, skeleton_config.right_parts, :] = tmpLeft

    return img, mask, meta
