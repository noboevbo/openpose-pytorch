import json
import os

import cv2

from network import get_network_results
from utils import util
from models.model_openpose import OpenPoseModelHandler
from skeletons.skeleton_config_openpose import SkeletonConfigOpenPose
from utils.util_eval import get_coco_joints_for_evaluation, get_coco_result_json, evaluate, get_result_json
from utils.util_predict import get_human_data


@util.measure_time
def get_result_json(model, img_path, image_id):
    original_img = cv2.imread(img_path)
    img_result = get_network_results(model, original_img)
    joint_positions, limbs, humans = get_human_data(img_result, original_img, skeleton_config)
    eval_humans = get_coco_joints_for_evaluation(humans, skeleton_config)
    json_data = get_coco_result_json(image_id, eval_humans)
    return json_data


if __name__ == "__main__":
    skeleton_config = SkeletonConfigOpenPose()
    val_img_folder = "/media/USERNAME/Data/Datasets/COCO/val2017"
    jsons = []
    count = 0
    files = os.listdir(val_img_folder)

    skeleton_config = SkeletonConfigOpenPose()
    model = OpenPoseModelHandler().get_model()
    OpenPoseModelHandler().load_state_dict(model)
    img_ids = []
    for file in files:
        if file.endswith(".jpg"):
            img_id = int(os.path.splitext(file)[0])
            img_ids.append(img_id)
            json_data = get_result_json(model, os.path.join(val_img_folder, file), img_id)
            jsons.extend(json_data)
            print("{}/{} ({}) (found: {})".format(count, len(files), file, len(json_data)))
            count += 1
            # if count > 25:
            #     break
    with open('/media/USERNAME/Data/Dump/test_exports/big_sim_c13.json', 'w') as outfile:
        json.dump(jsons, outfile)
    evaluate("/media/USERNAME/Data/Dump/test_exports/big_sim_c13.json")