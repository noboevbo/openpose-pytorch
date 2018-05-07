import concurrent.futures

import numpy as np
from scipy.ndimage import gaussian_filter

from utils import util
from config import cfg


@util.measure_time
def get_peaks(heatmap, joints):
    all_peaks = {}
    peak_counter = 0
    # execute the part calculation parallel len(joints) - 1 because background
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_joint_peaks, heatmap, part)
                   for part in range(len(joints))]
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            part, peaks = future.result()
            id = range(peak_counter, peak_counter + len(peaks))
            joint_peaks = []
            for idx, peak in enumerate(peaks):
                joint_peaks.append(
                    {'joint_nr': part, 'coords': [peak[0], peak[1]], 'x': peak[0], 'y': peak[1], 'score': peak[2],
                     'id': id[idx]})
            all_peaks[part] = joint_peaks
            peak_counter += len(joint_peaks)

    return all_peaks


def get_joint_peaks(average_joint_maps, part):
    part_heatmap_ori = average_joint_maps[:, :, part]
    part_heatmap = gaussian_filter(part_heatmap_ori, sigma=3)

    map_left = np.zeros(part_heatmap.shape)
    map_left[1:, :] = part_heatmap[:-1, :]
    map_right = np.zeros(part_heatmap.shape)
    map_right[:-1, :] = part_heatmap[1:, :]
    map_up = np.zeros(part_heatmap.shape)
    map_up[:, 1:] = part_heatmap[:, :-1]
    map_down = np.zeros(part_heatmap.shape)
    map_down[:, :-1] = part_heatmap[:, 1:]

    is_local_peak_list = np.logical_and.reduce(
        (part_heatmap >= map_left, part_heatmap >= map_right,
         part_heatmap >= map_up, part_heatmap >= map_down,
         part_heatmap > cfg.network.heatmap_thresh)
    )

    peaks = list(zip(np.nonzero(is_local_peak_list)[1], np.nonzero(is_local_peak_list)[0]))
    peaks_with_score = [x + (part_heatmap_ori[x[1], x[0]],) for x in peaks]

    return part, peaks_with_score
