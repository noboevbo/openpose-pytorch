import math
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np

from util_img import get_combined_maps, get_upscaled_maps


# Heatmap Plotting
def plot_map(in_map: np.ndarray, diverging: bool = False, is_255: bool = False):
    cmap = plt.cm.seismic if diverging else plt.cm.afmhot
    if is_255:
        plt.imshow(in_map, cmap=cmap, vmin=0, vmax=255)
    else:
        plt.imshow(in_map, cmap=cmap)
    plt.colorbar()


def plot_map_with_bg(original_img: np.ndarray, in_map: np.ndarray, diverging: bool = False, is_255: bool = False):
    cmap = plt.cm.seismic if diverging else plt.cm.afmhot
    plt.imshow(original_img, alpha=0.5)
    if is_255:
        plt.imshow(in_map, cmap=cmap, alpha=0.5, vmin=0, vmax=255)
    else:
        plt.imshow(in_map, cmap=cmap, alpha=0.5)
    plt.colorbar()


# Wrapper


#Variable to nd.array -> varname[0].data.cpu().numpy()
def visualize_network(original_img: np.ndarray, cropped_img: np.ndarray, joint_maps: np.ndarray, limb_maps: np.ndarray,
                      img_size: int, window_id=None):
    joint_map = get_combined_maps(joint_maps)
    limb_map = get_combined_maps(limb_maps)
    visualize_heatmaps(original_img, cropped_img, joint_map, limb_map, img_size, window_id)


# Network Outputs

def visualize_heatmaps(original_img: np.ndarray, input_img: np.ndarray, joint_map: np.ndarray, limb_map: np.ndarray,
                       img_size: int, window_id=None):
    fig = plt.figure(window_id, figsize=(11, 11), dpi=100)

    plt.suptitle('Results for img size: {}'.format(img_size))

    # Original Image
    fig_original = fig.add_subplot(2, 2, 1)
    fig_original.set_title('Original')
    plt.imshow(original_img)

    # Cropped Image
    fig_cropped = fig.add_subplot(2, 2, 2)
    fig_cropped.set_title('Input')
    plt.imshow(input_img)

    # Joint Map
    fig_joint_maps = fig.add_subplot(2, 2, 3)
    fig_joint_maps.set_title('Joint Maps')
    plot_map_with_bg(original_img, joint_map)

    # Limb Map
    fig_limb_maps = fig.add_subplot(2, 2, 4)
    fig_limb_maps.set_title('Limb Maps')
    plot_map_with_bg(original_img, limb_map)

    plt.show()


def build_compare_joint_limb_maps(joint_map_a: np.ndarray, limb_map_a: np.ndarray, joint_map_b: np.ndarray,
                                  limb_map_b: np.ndarray, bg_img=None, is_255: bool = False):
    def plt_func(in_map): plot_map_with_bg(bg_img, in_map, is_255=is_255) if bg_img is not None else plot_map(in_map, is_255=is_255)

    plt.cla()
    plt.clf()
    fig = plt.figure("cpr_maps", figsize=(11, 11), dpi=100)

    fig_joint_map = fig.add_subplot(2, 2, 1)
    fig_joint_map.set_title('Joint Map #1')
    plt_func(joint_map_a)

    fig_limb_map = fig.add_subplot(2, 2, 2)
    fig_limb_map.set_title('Limb Map #1')
    plt_func(limb_map_a)

    if joint_map_b is not None:
        fig_joint_map2 = fig.add_subplot(2, 2, 3)
        fig_joint_map2.set_title('Joint Map #2')
        plt_func(joint_map_b)

    if limb_map_b is not None:
        fig_limb_map2 = fig.add_subplot(2, 2, 4)
        fig_limb_map2.set_title('Limb Map #2')
        plt_func(limb_map_b)

    return fig


def visualize_compare_joint_limb_maps(joint_map_a: np.ndarray, limb_map_a: np.ndarray, joint_map_b: np.ndarray,
                                      limb_map_b: np.ndarray, bg_img=None):
    fig = build_compare_joint_limb_maps(joint_map_a, limb_map_a, joint_map_b, limb_map_b, bg_img)
    plt.draw()
    plt.pause(0.05)


def get_compare_joint_limb_maps(joint_map_a: np.ndarray, limb_map_a: np.ndarray, joint_map_b: np.ndarray,
                                limb_map_b: np.ndarray, bg_img=None):
    fig = build_compare_joint_limb_maps(joint_map_a, limb_map_a, joint_map_b, limb_map_b, bg_img)
    img_data = BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    file_bytes = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


# Visualize Limbs

def visualize_limbs(original_img: np.ndarray, joint_positions: [], limb_candidates: [], limbs: [], limb_colors: []):
    img = get_limb_image(original_img, joint_positions, limb_candidates, limbs, limb_colors)
    cv2.imshow("human_pose", img)
    cv2.waitKey(0)


def get_limb_image(original_img: np.ndarray, joint_positions: [], limb_candidates: [], limbs: [], limb_colors: []):
    canvas = original_img.copy()
    for i in range(18):
        for j in range(len(joint_positions[i])):
            cv2.circle(canvas, tuple(joint_positions[i][j]['coords']), 4, limb_colors[i], thickness=-1)

    stickwidth = 2

    for limb in limb_candidates:
        if limb['limb'] in [[2, 16], [5, 17]]:  # Ignore the Left/Right Eye to Left/Right Shoulder Joints
            continue
        color_idx = [i for i, x in enumerate(limbs) if x == limb['limb']]
        cur_canvas = canvas.copy()
        X = [limb['joint_a']['y'], limb['joint_b']['y']]
        Y = [limb['joint_a']['x'], limb['joint_b']['x']]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, limb_colors[color_idx[0]])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas


# Human Pose

def visualize_human_pose(original_img: np.ndarray, joint_positions: [], humans: [], limbs: [], limb_colors: []):
    img = get_human_pose_image(original_img, joint_positions, humans, limbs, limb_colors)
    cv2.imshow("human_pose", img)
    cv2.waitKey(0)


def save_human_pose_img(original_img: np.ndarray, joint_positions: [], humans: [], limbs: [], limb_colors: [],
                        file_path="human_pose.png"):
    img = get_human_pose_image(original_img, joint_positions, humans, limbs, limb_colors)
    cv2.imwrite(file_path, img)


def add_img_title(img, title):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title, (0, 30), font, 1, (0, 0, 255), 1, cv2.LINE_AA)


def get_human_pose_image(original_img: np.ndarray, joint_positions: [], humans: [], limbs: [], limb_colors: []):
    canvas = original_img.copy()
    add_img_title(canvas, "Humans: {}".format(len(humans)))
    for i in range(18):
        for j in range(len(joint_positions[i])):
            cv2.circle(canvas, tuple(joint_positions[i][j]['coords']), 4, limb_colors[i], thickness=1)

    stickwidth = 4

    for human in humans:
        for idx, limb in enumerate(human["limbs"]):
            if not limb:
                continue
            if limb['limb'] in [[2, 16], [5, 17]]:  # Ignore the Left/Right Eye to Left/Right Shoulder Joints
                continue
            # if limb['limb'] not in [[1,5], [1,2]]:
            #     continue
            color_idx = [i for i, x in enumerate(limbs) if x == limb['limb']]
            color = limb_colors[color_idx[0]]
            # color = [color[2], color[1], color[0]]
            #color = [0, 85, 255]
            print("{}: {}".format(limb['limb'], color))
            cur_canvas = canvas.copy()
            X = [limb['joint_a']['y'], limb['joint_b']['y']]
            Y = [limb['joint_a']['x'], limb['joint_b']['x']]
            cv2.circle(canvas, (limb['joint_a']['x'], limb['joint_a']['y']), 4, color, thickness=-1)
            cv2.circle(canvas, (limb['joint_b']['x'], limb['joint_b']['y']), 4, color, thickness=-1)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas


# Debug

def visualize_network_train_output(joint_maps_gt: np.array, limb_maps_gt: np.array, joint_maps: np.array,
                                   limb_maps: np.array, bg_img: np.array = None):
    """
    Opens some debug visualizations for network training outputs (compares with ground truth)
    """
    if bg_img is not None:
        img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        pad = [0, 0, 0, 0]
        upscaled_joint_maps = get_upscaled_maps(joint_maps_gt, img, img, pad)
        upscaled_limb_maps = get_upscaled_maps(limb_maps_gt, img, img, pad)
        upscaled_joint_maps_2 = get_upscaled_maps(joint_maps, img, img, pad)
        upscaled_limb_maps_2 = get_upscaled_maps(limb_maps, img, img, pad)
        visualize_compare_combined_joint_limb_maps(upscaled_joint_maps, upscaled_limb_maps, upscaled_joint_maps_2,
                                                   upscaled_limb_maps_2, img)
    else:
        visualize_compare_combined_joint_limb_maps(joint_maps_gt, limb_maps_gt,
                                                   joint_maps, limb_maps)


def visualize_compare_combined_joint_limb_maps(joint_maps_a: np.array, limb_maps_a: np.array, joint_maps_b: np.array,
                                               limb_maps_b: np.array, bg_img=None):
    joint_map = get_combined_maps(joint_maps_a)
    limb_map = get_combined_maps(limb_maps_a)
    joint_map2 = get_combined_maps(joint_maps_b)
    limb_map2 = get_combined_maps(limb_maps_b)
    visualize_compare_joint_limb_maps(joint_map, limb_map, joint_map2, limb_map2, bg_img)


def get_network_train_output(joint_maps_gt: np.array, limb_maps_gt: np.array, joint_maps: np.array,
                                   limb_maps: np.array, bg_img: np.array = None):
    """
    Opens some debug visualizations for network training outputs (compares with ground truth)
    """
    if bg_img is not None:
        img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        pad = [0, 0, 0, 0]
        upscaled_joint_maps = get_upscaled_maps(joint_maps_gt, img, img, pad)
        upscaled_limb_maps = get_upscaled_maps(limb_maps_gt, img, img, pad)
        upscaled_joint_maps_2 = None
        upscaled_limb_maps_2 = None
        if joint_maps is not None:
            upscaled_joint_maps_2 = get_upscaled_maps(joint_maps, img, img, pad)
        if limb_maps is not None:
            upscaled_limb_maps_2 = get_upscaled_maps(limb_maps, img, img, pad)
        return get_compare_combined_joint_limb_maps(upscaled_joint_maps, upscaled_limb_maps, upscaled_joint_maps_2,
                                                    upscaled_limb_maps_2, img)
    else:
        return get_compare_combined_joint_limb_maps(joint_maps_gt, limb_maps_gt, joint_maps, limb_maps)


def get_compare_combined_joint_limb_maps(joint_maps_a: np.array, limb_maps_a: np.array, joint_maps_b: np.array,
                                         limb_maps_b: np.array, bg_img=None):
    joint_map = get_combined_maps(joint_maps_a)
    limb_map = get_combined_maps(limb_maps_a)
    joint_map2 = None
    limb_map2 = None
    if joint_maps_b is not None:
        joint_map2 = get_combined_maps(joint_maps_b)
    if limb_maps_b is not None:
        limb_map2 = get_combined_maps(limb_maps_b)
    return get_compare_joint_limb_maps(joint_map, limb_map, joint_map2, limb_map2, bg_img)


def visualize_all_maps(maps: np.ndarray, bg_img: np.array = None, map_names: [] = None):
    def plt_func(in_map, in_img = None):
        plot_map_with_bg(in_img, in_map, is_255=True) if in_img is not None else plot_map(in_map, is_255=True)
    tmp_maps = maps
    num_maps = tmp_maps.shape[0]
    rows = math.ceil(num_maps / 4)
    fig = plt.figure("Debug Maps", figsize=(11, 11), dpi=100)

    for map_idx in range(num_maps):
        # Heatmap
        fig_heatmap = fig.add_subplot(rows, 4, map_idx + 1)
        map_name = map_names[map_idx] if map_names else map_idx
        fig_heatmap.set_title(map_name)
        tmp_map = np.array(tmp_maps[map_idx] * 255, dtype=np.uint8)
        plt_func(tmp_map, bg_img)

    plt.tight_layout()
    plt.show()


def visualize_map(map_data: np.array, bg_img: np.array = None, map_name: str = None):
    def plt_func(in_map, in_img = None):
        plot_map_with_bg(in_img, in_map) if in_img is not None else plot_map(in_map)
    tmp_map = np.array(map_data * 255, dtype=np.uint8)
    fig_name = map_name if map_name else "Debug Map"
    fig = plt.figure(fig_name, figsize=(11, 11), dpi=100)

    plt_func(tmp_map, bg_img)
    plt.tight_layout()
    plt.show()
