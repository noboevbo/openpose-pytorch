import json
import os

import cv2

from models.model_openpose import OpenPoseModelHandler as ModelHandler
from network import get_network_results
from skeletons.skeleton_config_openpose import SkeletonConfigOpenPose as SkeletonConfig
from utils import util
from utils.util_eval import get_joints_for_json_export, get_result_json
from utils.util_predict import get_human_data
from visualization import save_human_pose_img


@util.measure_time
def export(model, original_img, image_id):
    export_path = "/media/USERNAME/Data/rtpose2d/exports/rare_pose_dataset/big_sim_fixed_c13"
    img_result = get_network_results(model, original_img)
    joint_positions, limbs, humans = get_human_data(img_result, original_img, skeleton_config)
    humans_for_export = get_joints_for_json_export(humans, skeleton_config)
    export_json = get_result_json(image_id, humans_for_export)

    save_human_pose_img(original_img, joint_positions, humans, skeleton_config.limbs, skeleton_config.limb_colors,
                        file_path=os.path.join(export_path, "{}.jpg".format(image_id)))
    with open(os.path.join(export_path, "{}.json".format(image_id)), 'w') as outfile:
        json.dump(export_json, outfile)
    # save_human_pose_img(original_img, joint_positions, humans, SkeletonConfigRtPose2D.limbs, SkeletonConfigRtPose2D.limb_colors)


if __name__ == "__main__":
    skeleton_config = SkeletonConfig()
    jsons = []
    count = 0
    model_handler = ModelHandler()
    model = model_handler.get_model()
    model_handler.load_state_dict(model)
    dataset_path = "/media/USERNAME/Data/Datasets/rare_pose_dataset/images"
    for file in os.listdir(dataset_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_id = os.path.splitext(file)[0]
            print(count)
            image = cv2.imread(os.path.join(dataset_path, file))
            export(model, image, img_id)
            count += 1
