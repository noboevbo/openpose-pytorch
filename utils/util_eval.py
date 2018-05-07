import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from skeletons.joint_converteres.joint_converter_coco_rtpose2d import JointConverterCocoRtPose2D
from skeletons.skeleton_config_base import SkeletonConfigBase
from skeletons.skeleton_config_rtpose2d import SkeletonConfigRtPose2D

# TODO: Remove duplicated code with parametsrs (JointConfig and joints to ignore)


def get_joints_for_json_export(humans: {}, skeleton_config: SkeletonConfigBase):
    humans_joints = []
    for human in humans:
        human_joints = {}
        for limb_dict in human["limbs"]:
            if limb_dict is None:
                continue
            limb_joints = [limb_dict["joint_a"], limb_dict["joint_b"]]
            for joint in limb_joints:
                score = (limb_dict["matched_score"] + joint["score"]) / 2
                joint_name = skeleton_config.get_joint_name_by_id(joint["joint_nr"])
                joint = {
                    "id": SkeletonConfigRtPose2D.joints[joint_name],
                    "x": joint["x"],
                    "y": joint["y"],
                    "score": score
                }
                if joint["id"] not in human_joints:
                    human_joints[joint["id"]] = joint
        humans_joints.append(human_joints)
    return humans_joints


def get_result_json(image_id, humans: []):
    jsons_x = []
    for human in humans:
        json_data = {}
        json_data["image_id"] = image_id
        keypoint_list = []
        for i in range(len(SkeletonConfigRtPose2D.joints)):
            if i in human:
                joint = human[i]
                keypoint_list.append(int(joint["x"]))
                keypoint_list.append(int(joint["y"]))
                keypoint_list.append(int(joint["score"]))
            else:
                keypoint_list.append(0)
                keypoint_list.append(0)
                keypoint_list.append(0)
        json_data["keypoints"] = keypoint_list
        score = 0
        for joint in human.values():
            score += joint["score"]
        score = score / len(human)
        json_data["score"] = score

        jsons_x.append(json_data)
    return jsons_x


def get_coco_joints_for_evaluation(humans: {}, skeleton_config: SkeletonConfigBase):
    humans_joints = []
    for human in humans:
        human_joints = {}
        for limb_dict in human["limbs"]:
            if limb_dict is None:
                continue
            limb_joints = [limb_dict["joint_a"], limb_dict["joint_b"]]
            for joint in limb_joints:
                score = (limb_dict["matched_score"] + joint["score"]) / 2
                joint_name = skeleton_config.get_joint_name_by_id(joint["joint_nr"])
                if joint_name == "Neck":
                    continue #TODO: Handle better / generic. Neck is not in coco
                joint = {
                    "coco_id": JointConverterCocoRtPose2D.joints[joint_name],
                    "x": joint["x"],
                    "y": joint["y"],
                    "score": score
                }
                if joint["coco_id"] not in human_joints:
                    human_joints[joint["coco_id"]] = joint
        humans_joints.append(human_joints)
    return humans_joints


def get_coco_result_json(image_id, humans: []):
    jsons_x = []
    for human in humans:
        json_data = {}
        json_data["image_id"] = image_id
        json_data["category_id"] = 1
        keypoint_list = []
        for i in range(len(JointConverterCocoRtPose2D.joints)):
            if i in human:
                joint = human[i]
                keypoint_list.append(int(joint["x"]))
                keypoint_list.append(int(joint["y"]))
                keypoint_list.append(int(joint["score"]))
            else:
                keypoint_list.append(0)
                keypoint_list.append(0)
                keypoint_list.append(0)
        json_data["keypoints"] = keypoint_list
        score = 0
        for joint in human.values():
            score += joint["score"]
        score = score / len(human)
        json_data["score"] = score

        jsons_x.append(json_data)
    return jsons_x


def evaluate(result_json_file_path, img_ids_calced = None):
    annFile = '/media/USERNAME/Data/Datasets/COCO/annotations/person_keypoints_val2017.json'
    cocoGt = COCO(annFile)

    anns = json.load(open(result_json_file_path))
    annsImgIds = [ann['image_id'] for ann in anns]

    assert set(annsImgIds) == (set(annsImgIds) & set(cocoGt.getImgIds())), \
        'Results do not correspond to current coco set'

    resFile = result_json_file_path
    cocoDt = cocoGt.loadRes(resFile)

    #imgIds = imgIds[0:100]
    # imgId = imgIds[np.random.randint(5)]

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, "keypoints")
    cocoEval.params.imgIds = img_ids_calced if img_ids_calced else sorted(cocoGt.getImgIds())
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == "__main__":
    evaluate("/media/USERNAME/Data/rtpose2d/evaluations/caffe_openpose.json")