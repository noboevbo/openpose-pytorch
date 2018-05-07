from config import cfg
from skeletons.skeleton_config_rtpose2d import SkeletonConfigRtPose2D


def is_joint_from_limb_in_human(human, limb, limb_candidate):
    return human["joints"][limb[0]] == limb_candidate["joint_a"] or human["joints"][limb[1]] == limb_candidate["joint_b"]


def get_empty_human_dict(num_joints, num_limbs):
    human = {
        "score": 0,
        "num_joints": 0,
        "joints": [None] * num_joints,
        "limbs": [None] * num_limbs
    }
    return human


def are_joints_in_both_humans(human_a, human_b):
    for joint_idx, joint in enumerate(human_b["joints"]):
        if joint is not None and human_a["joints"][joint_idx] is not None:
            return True
    return False


def get_merged_humans(human_a, human_b):
    for joint_idx, joint in enumerate(human_b["joints"]):
        if joint is None:
            continue
        if human_a["joints"][joint_idx] is not None:
            raise RuntimeError("Merge conflict, joint exists in both humans")
        human_a["joints"][joint_idx] = joint

    for limb_idx, limb in enumerate(human_b["limbs"]):
        if limb is None:
            continue
        if human_a["limbs"][limb_idx] is not None:
            raise RuntimeError("Merge conflict, limb exists in both humans")#
        human_a["limbs"][limb_idx] = limb

    human_a["score"] += human_b["score"]
    human_a["num_joints"] += human_b["num_joints"]
    return human_a


def get_humans_from_limbs(limbs):
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    human_list = [] # Humans n, 20

    for limb_nr, limb_candidates in limbs.items():
        limb = SkeletonConfigRtPose2D.limbs[limb_nr]
        for i, limb_candidate in enumerate(limb_candidates):
            found = 0
            subset_idx = [-1, 1]
            for j in range(len(human_list)):
                if is_joint_from_limb_in_human(human_list[j], limb, limb_candidate):
                    subset_idx[found] = j
                    found += 1
            if found == 1:
                j = subset_idx[0]
                if human_list[j]["joints"][limb[1]] != limb_candidate["joint_b"]:
                    human_list[j]["joints"][limb[1]] = limb_candidate["joint_b"]
                    human_list[j]["limbs"][limb_candidate["limb_nr"]] = limb_candidate
                    human_list[j]["num_joints"] += 1
                    human_list[j]["score"] += limb_candidate["limb_score"] + limb_candidate["joint_b"]["score"]

            elif found == 2:
                j1, j2 = subset_idx
                #print("found = 2")
                if not are_joints_in_both_humans(human_list[j1], human_list[j2]):
                    human_list[j1] = get_merged_humans(human_list[j1], human_list[j2])
                    del human_list[j2]
                else:  # as like found == 1
                    human_list[j]["joints"][limb[1]] = limb_candidate["joint_b"]
                    human_list[j]["limbs"][limb_candidate["limb_nr"]] = limb_candidate
                    human_list[j]["num_joints"] += 1
                    human_list[j]["score"] += limb_candidate["limb_score"] + limb_candidate["joint_b"]["score"]
            elif not found:
                row = get_empty_human_dict(len(SkeletonConfigRtPose2D.joints), len(SkeletonConfigRtPose2D.limbs))
                row["joints"][limb[0]] = limb_candidate["joint_a"]
                row["joints"][limb[1]] = limb_candidate["joint_b"]
                row["limbs"][limb_candidate["limb_nr"]] = limb_candidate
                row["num_joints"] = 2
                row["score"] = limb_candidate["matched_score"]
                human_list.append(row)

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(human_list)):
        if human_list[i]["num_joints"] < cfg.network.skeleton_min_limbs or human_list[i]["score"] / \
                human_list[i]["num_joints"] < cfg.network.skeleton_limb_score:
            deleteIdx.append(i)
    return [x for i, x in enumerate(human_list) if i not in deleteIdx]
