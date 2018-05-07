import os

import cv2
import numpy as np
import torch
from scipy.io import loadmat

from config import cfg
from datasets.Augmenter import AugmentSelection, transform
from datasets.dataset_base import DatasetBase
from models.model_base import NetworkModelBase
from skeletons.gt_generators.gt_generator_base import GroundTruthGeneratorBase
from skeletons.joint_converteres.joint_converter_lsp_rtpose2d import JointConverterLspRtPose2D
from skeletons.skeleton_config_base import SkeletonConfigBase
from util_img import normalize


class LspDataset(DatasetBase):
    # Format: x, y, visible, visible->0 == visible, 1 == invisible
    def __init__(self, lsp_dataset_path, skeleton_config: SkeletonConfigBase, gt_generator: GroundTruthGeneratorBase,
                 model: NetworkModelBase, num_samples=None, augment=False):
        self.path = lsp_dataset_path
        self.skeleton_config = skeleton_config
        self.augment = augment
        self.model = model
        self.gt_generator = gt_generator
        self.image_folder = os.path.join(self.path, "images")
        self.joint_converter = JointConverterLspRtPose2D()

        joints = loadmat(os.path.join(self.path, "joints.mat"))
        joints = joints['joints'].transpose(2, 1, 0)

        keys = list(range(0, joints.shape[0]))
        if num_samples:
            keys = np.random.choice(keys, num_samples, replace=False).tolist()
        self.items = []
        for key in keys:
            self.items.append({"lsp_id": key + 1, "joints": joints[key, :, :]})

    def __getitem__(self, index):
        item = self.items[index]
        img_path = os.path.join(self.image_folder, "im{0:04d}.jpg".format(item["lsp_id"]))
        image = cv2.imread(img_path)
        meta = {"scale_provided": [150 / cfg.general.input_height],
                "joints": self.joint_converter.get_converted_joint_list(item["joints"])}
        rhip_pos = meta["joints"][0, 8, 0:2]
        lhip_pos = meta["joints"][0, 11, 0:2]
        meta["objpos"] = [[(rhip_pos[0] + lhip_pos[0]) / 2, (rhip_pos[1] + lhip_pos[1]) / 2]]
        mask_misss = np.ones((image.shape[0], image.shape[1])) * 255
        image, mask_misss, meta, labels = self.transform_data(image, mask_misss, meta)
        image = self.get_img_as_tensor(image)
        limb_map_masks = self.get_mask_as_tensor(np.repeat(mask_misss[:, :, np.newaxis], self.model.num_limb_maps, axis=2))
        joint_map_masks = self.get_mask_as_tensor(np.repeat(mask_misss[:, :, np.newaxis], self.model.num_joint_maps, axis=2))

        limb_maps = torch.from_numpy(labels[:self.model.num_limb_maps, :, :]).float()
        joint_maps = torch.from_numpy(labels[self.model.num_limb_maps:, :, :]).float()
        return {'image': image, 'joint_map_gt': joint_maps, 'limb_map_gt': limb_maps,
                'limb_map_masks': limb_map_masks, 'joint_map_masks': joint_map_masks}

    def __len__(self):
        return len(self.items)

    def get_dataset_id_from_index(self, index):
        item = self.items[index]
        return int(item["lsp_id"])

    def get_img_as_tensor(self, img):
        image = np.transpose(img, (2, 0, 1))  # transpose to channels, height, width
        image = normalize(image)
        return torch.from_numpy(image).float()

    def get_mask_as_tensor(self, img):
        image = np.transpose(img, (2, 0, 1))  # transpose to channels, height, width
        return torch.from_numpy(image).float()

    def transform_data(self, img, mask, meta):
        aug = AugmentSelection.random() if self.augment else AugmentSelection.unrandom()
        img, mask, meta = transform(img, mask, meta, self.model.mask_shape, self.skeleton_config, aug=aug)
        labels = self.gt_generator.get_ground_truth(meta['joints'], mask)

        return img, mask, meta, labels
