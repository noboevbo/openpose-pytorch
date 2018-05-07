import numpy as np

from skeletons.joint_converteres.joint_converter_base import JointConverterBase
from skeletons.skeleton_config_rtpose2d import SkeletonConfigRtPose2D


def get_internal_neck_visibility(joints, both_shoulders_known, r_shoulder, l_shoulder):
    return np.minimum(joints[both_shoulders_known, r_shoulder, 2], joints[both_shoulders_known, l_shoulder, 2])


def get_internal_neck_position(joints, both_shoulders_known, r_shoulder, l_shoulder):
    return (joints[both_shoulders_known, r_shoulder, 0:2] + joints[both_shoulders_known, l_shoulder, 0:2]) / 2


class JointConverterCocoRtPose2D(JointConverterBase):
    joints = {
        'Nose': 0,
        'LEye': 1,
        'REye': 2,
        'LEar': 3,
        'REar': 4,
        'LShoulder': 5,
        'RShoulder': 6,
        'LElbow': 7,
        'RElbow': 8,
        'LWrist': 9,
        'RWrist': 10,
        'LHip': 11,
        'RHip': 12,
        'LKnee': 13,
        'RKnee': 14,
        'LAnkle': 15,
        'RAnkle': 16
    }

    def get_converted_joint_list(self, source_joints: []) -> np.array:
        joints = np.array(source_joints)
        result = np.zeros((joints.shape[0], len(SkeletonConfigRtPose2D.joints), 3), dtype=np.float)
        result[:, :, 2] = 2.  # 2 - absent, 1 visible, 0 - invisible

        for coco_joint, coco_joint_id in self.joints.items():
            internal_joint_id = SkeletonConfigRtPose2D.joints[coco_joint]
            assert internal_joint_id != 1, "Neck shouldn't be known yet"
            result[:, internal_joint_id, :] = joints[:, coco_joint_id, :]

        neck_internal = SkeletonConfigRtPose2D.joints['Neck']
        r_shoulder_coco = self.joints['RShoulder']
        l_shoulder_coco = self.joints['LShoulder']

        # no neck in coco database, we calculate it as average of shoulders
        # TODO: we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
        both_shoulders_known = (joints[:, l_shoulder_coco, 2] < 2) & (joints[:, r_shoulder_coco, 2] < 2)
        result[both_shoulders_known, neck_internal, 0:2] = get_internal_neck_position(joints, both_shoulders_known,
                                                                                           r_shoulder_coco, l_shoulder_coco)
        result[both_shoulders_known, neck_internal, 2] = get_internal_neck_visibility(joints, both_shoulders_known,
                                                                                           r_shoulder_coco, l_shoulder_coco)

        return result
