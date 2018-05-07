import numpy as np


class JointConverterBase:
    def get_converted_joint_list(self, source_joints: []) -> np.array:
        raise NotImplementedError