#!/usr/bin/env python

from math import sqrt, isnan

import numpy as np

from config import cfg
from models.model_openpose import OpenPoseModel
from skeletons.gt_generators.gt_generator_base import GroundTruthGeneratorBase
from skeletons.skeleton_config_base import SkeletonConfigBase


class GroundTruthGeneratorOpenPose(GroundTruthGeneratorBase):
    def __init__(self, model: OpenPoseModel, skeleton_config: SkeletonConfigBase):
        self.double_sigma2 = 2 * cfg.train.augmentation.sigma * cfg.train.augmentation.sigma
        self.thre = cfg.network.paf_thre
        self.model = model
        self.skeleton_config = skeleton_config

        # cached common parameters which same for all iterations and all pictures

        stride = cfg.general.stride
        width = cfg.general.input_width//stride
        height = cfg.general.input_height//stride

        # this is coordinates of centers of bigger grid
        self.grid_x = np.arange(width)*stride + stride/2-0.5
        self.grid_y = np.arange(height)*stride + stride/2-0.5

        self.Y, self.X = np.mgrid[0:cfg.general.input_height:stride,0:cfg.general.input_width:stride]

    def get_ground_truth(self, joints, mask_miss):
        heatmaps = np.zeros(self.model.parts_shape, dtype=np.float)

        self.generate_joint_heatmaps(heatmaps, joints)
        slice_joint_maps = slice(self.model.joint_maps_start, self.model.joint_map_bg)
        heatmaps[self.model.joint_map_bg] = 1. - np.amax(heatmaps[slice_joint_maps, :, :], axis=0)

        self.generate_limb_pafs(heatmaps, joints)

        heatmaps *= mask_miss

        return heatmaps

    def put_gaussian_maps(self, heatmaps, layer, joints):
        # actually exp(a+b) = exp(a)*exp(b), lets use it calculating 2d exponent, it could just be calculated by
        for i in range(joints.shape[0]):

            exp_x = np.exp(-(self.grid_x-joints[i,0])**2/self.double_sigma2)
            exp_y = np.exp(-(self.grid_y-joints[i,1])**2/self.double_sigma2)

            exp = np.outer(exp_y, exp_x)

            # note this is correct way of combination - min(sum(...),1.0) as was in C++ code is incorrect
            # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/118
            heatmaps[self.model.joint_maps_start + layer, :, :] = \
                np.maximum(heatmaps[self.model.joint_maps_start + layer, :, :], exp)

    def generate_joint_heatmaps(self, heatmaps, joints):
        for i in range(len(self.skeleton_config.joints)):
            # 0 - invisible, 1 visible, 2 - absent
            joint_visibility_flags = joints[:,i,2]
            visible = joint_visibility_flags < 2
            self.put_gaussian_maps(heatmaps, i, joints[visible, i, 0:2])

    def put_vector_maps(self, heatmaps, layerX, layerY, joint_from, joint_to):
        count = np.zeros(heatmaps.shape[1:], dtype=np.int)

        for i in range(joint_from.shape[0]):
            (x1, y1) = joint_from[i]
            (x2, y2) = joint_to[i]

            dx = x2-x1
            dy = y2-y1
            dnorm = sqrt(dx*dx + dy*dy)

            if dnorm==0:  # we get nan here sometimes, it's kills NN
                # TODO: handle it better. probably we should add zero paf, centered paf, or skip this completely
                print("Parts are too close to each other. Length is zero. Skipping")
                continue

            dx = dx / dnorm
            dy = dy / dnorm

            assert not isnan(dx) and not isnan(dy), "dnorm is zero, wtf"

            min_sx, max_sx = (x1, x2) if x1 < x2 else (x2, x1)
            min_sy, max_sy = (y1, y2) if y1 < y2 else (y2, y1)

            stride = cfg.general.stride
            min_sx = int(round((min_sx - self.thre) / stride))
            min_sy = int(round((min_sy - self.thre) / stride))
            max_sx = int(round((max_sx + self.thre) / stride))
            max_sy = int(round((max_sy + self.thre) / stride))

            # check PAF off screen. do not really need to do it with max>grid size
            if max_sy < 0:
                continue

            if max_sx < 0:
                continue

            if min_sx < 0:
                min_sx = 0

            if min_sy < 0:
                min_sy = 0

            slice_x = slice(min_sx, max_sx) # + 1     this mask is not only speed up but crops paf really. This copied from original code
            slice_y = slice(min_sy, max_sy) # + 1     int g_y = min_y; g_y < max_y; g_y++ -- note strict <

            dist = distances(self.X[slice_y,slice_x], self.Y[slice_y,slice_x], x1, y1, x2, y2)
            dist = dist <= self.thre

            heatmaps[layerX, slice_y, slice_x][dist] = (dist * dx)[dist]  # += dist * dx
            heatmaps[layerY, slice_y, slice_x][dist] = (dist * dy)[dist] # += dist * dy
            count[slice_y, slice_x][dist] += 1

        # TODO: averaging by pafs mentioned in the paper but never worked in C++ augmentation code
        # heatmaps[layerX, :, :][count > 0] /= count[count > 0]
        # heatmaps[layerY, :, :][count > 0] /= count[count > 0]

    def generate_limb_pafs(self, heatmaps, joints):
        for (i,(fr,to)) in enumerate(self.skeleton_config.limbs): # TODO Check if this works, tuple instead list
            visible_from = joints[:,fr,2] < 2
            visible_to = joints[:,to, 2] < 2
            visible = visible_from & visible_to

            # get from mapping
            mapping = self.model.limb_paf_mapping[i]
            paf_layer_x, paf_layer_y = mapping[0], mapping[1]
            self.put_vector_maps(heatmaps, paf_layer_x, paf_layer_y, joints[visible, fr, 0:2], joints[visible, to, 0:2])


def distances(X, Y, x1, y1, x2, y2):

    # classic formula is:
    # d = (x2-x1)*(y1-y)-(x1-x)*(y2-y1)/sqrt((x2-x1)**2 + (y2-y1)**2)

    xD = (x2-x1)
    yD = (y2-y1)
    norm2 = sqrt(xD**2 + yD**2)
    dist = xD*(y1-Y)-(x1-X)*yD
    dist /= norm2

    return np.abs(dist)
