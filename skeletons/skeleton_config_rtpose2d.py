from skeletons.joint_setup import joints_rtpose2d
from skeletons.skeleton_config_base import SkeletonConfigBase
from collections import defaultdict


def ltr_parts(parts_dict):
    # when we flip image left parts became right parts and vice versa. This is the list of parts to exchange each other.
    left_parts = [parts_dict[p] for p in
                  ["LShoulder", "LElbow", "LWrist", "LHip", "LKnee", "LAnkle", "LEye", "LEar"]]
    right_parts = [parts_dict[p] for p in
                   ["RShoulder", "RElbow", "RWrist", "RHip", "RKnee", "RAnkle", "REye", "REar"]]
    return left_parts, right_parts


class SkeletonGraphRtPose2D:
    def __init__(self):
        self.outgoing_limbs = defaultdict(list)
        self.incoming_limbs = defaultdict(list)
        self.outgoing_joints = defaultdict(list)
        self.incoming_joints = defaultdict(list)

        self.edges = defaultdict(list)
        self.weights = {}
        self.joint_instances = defaultdict(list)

    def add_instances(self, limbs):
        for limb in limbs:
            joint_a = limb["joint_a"]
            joint_a = (joint_a["joint_nr"], joint_a["x"], joint_a["y"])
            joint_b = limb["joint_b"]
            joint_b = (joint_b["joint_nr"], joint_b["x"], joint_b["y"])
            self.edges[joint_a].append(joint_b)
            self.weights[(joint_a, joint_b)] = limb["matched_score"]
            if joint_a not in self.joint_instances[joint_a[0]]:
                self.joint_instances[joint_a[0]].append(joint_a)
            if joint_b not in self.joint_instances[joint_b[0]]:
                self.joint_instances[joint_b[0]].append(joint_b)

    def neighbors(self, id):
        return self.edges[id]

    def cost(self, from_node, to_node):
        return self.weights[(from_node, to_node)]


class SkeletonConfigRtPose2D(SkeletonConfigBase):
    important_joints = []
    important_limbs = [6, 7, 8, 9, 10, 11] # Right and Left Neck -> Ankle Connection

    joints = joints_rtpose2d

    limbs = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
             [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
             [0, 15], [15, 17], [2, 16], [5, 17]]

    joint_is_source_in_limb = defaultdict(list)
    joint_is_target_in_limb = defaultdict(list)
    for joint_name, joint_num in joints.items():
        for limb_nr, limb in enumerate(limbs):
            if limb[0] == joint_num:
                joint_is_source_in_limb[joint_num].append(limb_nr)
            if limb[1] == joint_num:
                joint_is_target_in_limb[joint_num].append(limb_nr)

    graph = SkeletonGraphRtPose2D()
    for joint_name, joint_num in joints.items():
        for limb_nr, limb in enumerate(limbs):
            if limb[0] == joint_num:
                graph.outgoing_limbs[joint_name].append(limb)
                graph.outgoing_joints[joint_name].append(list(joints.items())[limb[1]])
            if limb[1] == joint_num:
                graph.incoming_limbs[joint_name].append(limb)
                graph.incoming_joints[joint_name].append(list(joints.items())[limb[0]])
    a = 1

    limb_names = []
    __joint_list = list(joints.items())
    for limb in limbs:
        limb_names.append("{}-{}".format(__joint_list[limb[0]][0], __joint_list[limb[1]][0]))
    limb_names.append("Background")

    limb_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                   [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                   [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [0, 0, 0]]

    left_parts, right_parts = ltr_parts(joints)