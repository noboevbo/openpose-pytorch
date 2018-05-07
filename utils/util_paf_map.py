import math
from collections import defaultdict

import numpy as np

from utils import util
from config import cfg


@util.measure_time
def get_limbs(paf_results, joint_positions, original_img, limb_cfg, limb_paf_mapping):
    limbs_all = defaultdict(list)

    for k in range(len(limb_paf_mapping)):
        score_mid = paf_results[:, :, [x for x in limb_paf_mapping[k]]]
        candidates_a = joint_positions[limb_cfg[k][0]]
        candidates_b = joint_positions[limb_cfg[k][1]]
        num_candidates_a = len(candidates_a)
        num_candidates_b = len(candidates_b)
        if num_candidates_a != 0 and num_candidates_b != 0:
            limb_candidates = get_limb_candidates(candidates_a, candidates_b, score_mid,
                                                  original_img)

            limbs = 0
            used_joint_as = []
            used_joint_bs = []
            for limb_candidate in limb_candidates:
                if limb_candidate['joint_a_idx'] not in used_joint_as and limb_candidate['joint_b_idx'] not in used_joint_bs:
                    used_joint_as.append(limb_candidate['joint_a_idx'])
                    used_joint_bs.append(limb_candidate['joint_b_idx'])
                    limbs_all[k].append({
                        'limb_nr': k,
                        'limb': limb_cfg[k],
                        'joint_a': limb_candidate['joint_a'],
                        'joint_b': limb_candidate['joint_b'],
                        'limb_score': limb_candidate['limb_score'],
                        'matched_score': limb_candidate['matched_score']
                    })
                    limbs += 1
                    if limbs >= min(num_candidates_a, num_candidates_b):
                        break
        else:
            # TODO: Handle limb Ks missing joints somehow?
            continue
    return limbs_all


def get_limb_candidates(candidates_a, candidates_b, score_mid, original_img):
    """
    Returns limb candidates between (joint) candidates_a and (joint) candidates_b
    """
    limb_candidates = []
    for i in range(len(candidates_a)):
        for j in range(len(candidates_b)):
            joint_a = candidates_a[i]['coords']
            joint_b = candidates_b[j]['coords']
            paf_x = np.squeeze(score_mid[:, :, :1], axis=2)
            paf_y = np.squeeze(score_mid[:, :, 1:], axis=2)

            limb_score, sample_scores = get_limb_score(original_img, joint_a, joint_b, paf_x, paf_y)
            if limb_score <= 0:
                continue
            samples_over_thresh = np.nonzero(sample_scores > cfg.network.paf_thresh_sample_score)[0]
            enough_samples_over_thresh = len(samples_over_thresh) > cfg.network.paf_samples_over_thresh * len(
                sample_scores)
            min_joint_score_reached = limb_score > 0
            if enough_samples_over_thresh and min_joint_score_reached:
                limb_candidates.append({
                    'joint_a_idx': i,
                    'joint_b_idx': j,
                    'joint_a': candidates_a[i],
                    'joint_b': candidates_b[j],
                    'limb_score': limb_score,
                    'matched_score': limb_score + candidates_a[i]['score'] + candidates_b[j]['score']
                }
                )
    return sorted(limb_candidates, key=lambda x: x['limb_score'], reverse=True)


def get_limb_score(img, point_a, point_b, paf_x, paf_y):
    """
    Calculates a score for a limb between the given points p1 and p2. Score is calculated by the line integral which
    measures the effect of the part affinity fields along the given joint.
    """
    num_samples = cfg.network.paf_num_samples
    x1, y1 = point_a[0], point_a[1]
    x2, y2 = point_b[0], point_b[1]

    distance_x, distance_y = x2 - x1, y2 - y1
    distance_joints = math.sqrt(distance_x ** 2 + distance_y ** 2)

    if distance_joints < 1e-4:
        return 0.0, np.zeros([10], dtype=np.float32)

    vx, vy = distance_x / distance_joints, distance_y / distance_joints
    xs = np.around(np.linspace(x1, x2, num=num_samples)).astype(np.uint32)
    ys = np.around(np.linspace(y1, y2, num=num_samples)).astype(np.uint32)

    paf_xs = np.zeros(num_samples)
    paf_ys = np.zeros(num_samples)
    for idx, (mx, my) in enumerate(zip(xs, ys)):
        paf_xs[idx] = paf_x[my][mx]
        paf_ys[idx] = paf_y[my][mx]

    sample_scores = paf_xs * vx + paf_ys * vy
    d_punishment = min(0.5 * img.shape[0] / distance_joints - 1, 0)  # Punish joint distances > img_height/2
    line_integral = sum(sample_scores) / num_samples + d_punishment
    return line_integral, sample_scores

# Ground Truth
