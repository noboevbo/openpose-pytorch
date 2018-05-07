from collections import OrderedDict

import numpy as np

from skeletons.joint_converteres.joint_converter_base import JointConverterBase
from skeletons.skeleton_config_rtpose2d import SkeletonConfigRtPose2D


class JointConverterLspRtPose2D(JointConverterBase):
    joints = OrderedDict([
        ("Nose", None),
        ("Neck", 12),
        ("RShoulder", 8),
        ("RElbow", 7),
        ("RWrist", 6),
        ("LShoulder", 9),
        ("LElbow", 10),
        ("LWrist", 11),
        ("RHip", 2),
        ("RKnee", 1),
        ("RAnkle", 0),
        ("LHip", 3),
        ("LKnee", 4),
        ("LAnkle", 5),
        ("REye", None),
        ("LEye", None),
        ("REar", None),
        ("LEar", None)
    ])

    # LSP Visibility: 0 = Visible, 1 = Invisible
    # Always only one annotated person, thus 1 as array rows
    def get_converted_joint_list(self, source_joints: []) -> np.array:
        joints = np.array(source_joints)
        result = np.zeros((1, len(SkeletonConfigRtPose2D.joints), 3), dtype=np.float)
        result[:, :, 2] = 2.  # 2 - absent, 1 visible, 0 - invisible

        for rtpose2d_joint, rtpose2d_joint_id in SkeletonConfigRtPose2D.joints.items():
            joint_num = self.joints[rtpose2d_joint]
            if joint_num:
                result[:, rtpose2d_joint_id, :] = np.expand_dims(joints[joint_num], axis=0)
                visibility_lsp = result[:, rtpose2d_joint_id, 2]
                result[:, rtpose2d_joint_id, 2] = 1 - visibility_lsp

        return result
