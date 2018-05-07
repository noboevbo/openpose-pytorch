from skeletons.skeleton_config_rtpose2d import SkeletonConfigRtPose2D


class SkeletonConfigOpenPose(SkeletonConfigRtPose2D):
    def __init__(self):
        self.limb_names = []
        __joint_list = list(self.joints.items())
        for limb in self.limbs:
            self.limb_names.append("{}-{}-X".format(__joint_list[limb[0]][0], __joint_list[limb[1]][0]))
            self.limb_names.append("{}-{}-Y".format(__joint_list[limb[0]][0], __joint_list[limb[1]][0]))
        self.limb_names.append("Background-X")
        self.limb_names.append("Background-Y")
