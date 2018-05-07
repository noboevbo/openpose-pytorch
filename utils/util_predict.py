import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from models.model_base import NetworkModelBase
from skeletons.skeleton_config_base import SkeletonConfigBase
from utils import util, util_paf_map
from utils.util_joint_map import get_peaks
from utils.util_skeleton import get_humans_from_limbs


def get_avg_map_mulpr(img_result):
    upscaled_limb_maps = util.get_upsampled_maps(img_result, "limb_maps")
    return util.get_average_map_from_upscaled_maps(upscaled_limb_maps)


def get_human_data(model: NetworkModelBase, img_result: dict, original_img: np.ndarray, skeleton_config: SkeletonConfigBase):
    with ThreadPoolExecutor() as executor:
        future = executor.submit(get_avg_map_mulpr, img_result)
        start_time = time.time()
        upscaled_joint_maps = util.get_upsampled_maps(img_result, "joint_maps")
        average_joint_maps = util.get_average_map_from_upscaled_maps(upscaled_joint_maps)
        print("{}: {}".format("Upscale joints", time.time() - start_time))

        joint_positions = get_peaks(average_joint_maps, skeleton_config.joints)

        average_limb_maps = future.result()
    print("{}: {}".format("Upscale Limbs / Get Peaks", time.time() - start_time))

    limbs = util_paf_map.get_limbs(average_limb_maps, joint_positions, original_img, skeleton_config.limbs,
                                   model.limb_paf_mapping)
    humans = get_humans_from_limbs(limbs)
    return joint_positions, limbs, humans