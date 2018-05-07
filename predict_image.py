import os

import cv2

import config
from network import get_network_results
from utils import util
from models.model_openpose import OpenPoseModelHandler
from skeletons.skeleton_config_openpose import SkeletonConfigOpenPose
from utils.util_predict import get_human_data
from visualization import visualize_human_pose, save_human_pose_img

skeleton_config = SkeletonConfigOpenPose()


@util.measure_time
def predict(img_path, visualize=False):
    model = OpenPoseModelHandler().get_model()
    OpenPoseModelHandler().load_state_dict(model)
    original_img = cv2.imread(img_path)
    img_result = get_network_results(model, original_img)
    joint_positions, limbs, humans = get_human_data(model, img_result, original_img, skeleton_config)

    if visualize:
        visualize_human_pose(original_img, joint_positions, humans, skeleton_config.limbs, skeleton_config.limb_colors)
    # save_human_pose_img(original_img, joint_positions, humans, skeleton_config.limbs, skeleton_config.limb_colors)


if __name__ == "__main__":
    config.cfg = config.OpenPoseConfig()
    predict("/home/USERNAME/Pictures/test.jpg", True)