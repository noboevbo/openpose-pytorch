import json
import logging

import cv2
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp

from datasets.Augmenter import AugmentSelection, transform
from datasets.dataset_base import DatasetBase
from models.model_base import NetworkModelBase
from skeletons.gt_generators.gt_generator_base import GroundTruthGeneratorBase
from skeletons.joint_converteres.joint_converter_coco_rtpose2d import JointConverterCocoRtPose2D
from skeletons.skeleton_config_base import SkeletonConfigBase
from skeletons.skeleton_config_openpose import SkeletonConfigOpenPose
from skeletons.skeleton_config_rtpose2d import SkeletonConfigRtPose2D
from util_img import normalize


def get_joint_converter(skeleton_config):
    if type(skeleton_config) is SkeletonConfigRtPose2D:
        return JointConverterCocoRtPose2D()
    if type(skeleton_config) is SkeletonConfigOpenPose:
        return JointConverterCocoRtPose2D()


class CocoDataset(DatasetBase):
    def __init__(self, h5files: [str], skeleton_config: SkeletonConfigBase, gt_generator: GroundTruthGeneratorBase,
                 model: NetworkModelBase, num_samples=None, augment=False):
        self.h5s = [h5py.File(fname, "r") for fname in h5files]
        self.h5_contents = [(h5['dataset'], h5['images'], h5['miss_masks'] if 'miss_masks' in h5 else None) for h5 in
                            self.h5s]
        self.joint_converter = get_joint_converter(skeleton_config)
        self.skeleton_config = skeleton_config
        self.keys = []
        self.lock = mp.Lock()
        self.augment = augment
        self.model = model
        self.gt_generator = gt_generator

        self.logger = logging.getLogger("coco_dataset_log")
        self.hdlr = logging.FileHandler('coco_dataset.log')
        self.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        self.hdlr.setFormatter(self.formatter)
        self.logger.addHandler(self.hdlr)
        self.logger.setLevel(logging.INFO)

        with self.lock:
            for idx, content in enumerate(self.h5_contents):
                keys = list(content[0].keys())
                a = len(keys)
                if num_samples:
                    keys = np.random.choice(keys, num_samples, replace=False).tolist()
                b = len(keys)
                #print(len(keys))

                self.keys += zip([idx] * len(keys), keys)

    def __getitem__(self, index):
        key = self.keys[index]
        self.logger.info("IDX[{}], KEY[{}]".format(index, key))
        image, mask_misss, meta = self.read_data(key[0], key[1])
        if image is None or image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] != 3:
            self.logger.error("Error with img idx: {}, key: {}".format(index, key))
            idx_to_use = index+1
            if idx_to_use >= self.__len__():
                idx_to_use = 0
            return self.__getitem__(idx_to_use)
        image, mask_misss, meta, labels = self.transform_data(image, mask_misss, meta)
        image = self.get_img_as_tensor(image)
        limb_map_masks = self.get_mask_as_tensor(np.repeat(mask_misss[:, :, np.newaxis], self.model.num_limb_maps, axis=2))
        joint_map_masks = self.get_mask_as_tensor(np.repeat(mask_misss[:, :, np.newaxis], self.model.num_joint_maps, axis=2))

        limb_maps = torch.from_numpy(labels[:self.model.num_limb_maps, :, :]).float()
        joint_maps = torch.from_numpy(labels[self.model.num_limb_maps:, :, :]).float()
        return {'image': image, 'joint_map_gt': joint_maps, 'limb_map_gt': limb_maps,
                'limb_map_masks': limb_map_masks, 'joint_map_masks': joint_map_masks}

    def __len__(self):
        return len(self.keys)

    def get_dataset_id_from_index(self, index):
        key = self.keys[index]
        image, mask_misss, meta = self.read_data(key[0], key[1])
        return int(meta["image"])

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

    def read_data(self, num, key):
        content = self.h5_contents[num]
        dataset, images, mask_misss = content
        return self.read_data_new(dataset, images, mask_misss, key)

    def read_data_new(self, dataset, images, mask_misss, key):
        with self.lock: #HDF5 is not threadsafe, so lock while accessing it.
            entry = dataset[key]
            meta = json.loads(entry.value)
            img = images[meta['image']].value
            mask_miss = mask_misss[meta['image']].value
            #debug = json.loads(entry.attrs['meta'])
        meta["joints"] = self.joint_converter.get_converted_joint_list(meta['joints'])

        img = cv2.imdecode(img, flags=-1)
        mask_miss = cv2.imdecode(mask_miss, flags=-1)  # TODO: Mask_Miss always available?

        return img, mask_miss, meta
