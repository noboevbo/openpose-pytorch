from collections import OrderedDict


class SkeletonConfigBase:
    @property
    def joints(self) -> OrderedDict:
        raise NotImplementedError

    @property
    def limbs(self) -> [[int, int]]:
        raise NotImplementedError

    @property
    def left_parts(self) -> [str]:
        raise NotImplementedError

    @property
    def right_parts(self) -> [str]:
        raise NotImplementedError

    @property
    def important_limbs(self) -> [int]:
        raise NotImplementedError

    @property
    def important_joints(self) -> [int]:
        raise NotImplementedError

    def get_joint_name_by_id(self, joint_id):
        return list(self.joints.items())[joint_id][0]