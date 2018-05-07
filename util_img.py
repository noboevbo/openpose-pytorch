import cv2
import numpy as np

from config import cfg


def normalize(img: np.array):
    img = img / 255.0
    img = img - 0.5
    return img


def denormalize(img: np.array):
    img = img + 0.5
    img *= 255.0
    return img


def get_img_from_network_output(img: np.ndarray) -> np.ndarray:
    img = denormalize(img)
    img = img.astype(np.uint8)
    return np.transpose(img, (1, 2, 0))


def get_combined_maps(maps: np.ndarray) -> np.ndarray:
    combined_map = np.transpose(maps, (1, 2, 0))
    return np.amax(combined_map, axis=2)


def get_upscaled_map(map_data: np.ndarray, original_img: np.ndarray, input_img: np.ndarray, pad: []) -> np.ndarray:
    tmp_map = cv2.resize(map_data, (0, 0), fx=cfg.general.stride, fy=cfg.general.stride, interpolation=cv2.INTER_CUBIC)
    tmp_map = tmp_map[:input_img.shape[0] - pad[2], :input_img.shape[1] - pad[3]]
    return cv2.resize(tmp_map, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)


def get_upscaled_maps(maps: np.ndarray, original_img: np.ndarray, input_img: np.ndarray, pad: []) -> np.ndarray:
    num_maps = maps.shape[0]
    output_maps = np.empty((num_maps, cfg.general.input_width, cfg.general.input_height))
    for i in range(num_maps):
        output_maps[i] = get_upscaled_map(maps[i, :, :], original_img, input_img, pad)
    return output_maps


def get_img_padded_as_box(img: np.ndarray):
    new_img = img.copy()
    if img.shape[0] == img.shape[1]:
        return new_img
    elif img.shape[0] > img.shape[1]:
        size_diff = img.shape[0] - img.shape[1]
        top = 0
        bottom = 0
        left = 0
        right = 0
        if size_diff % 2 == 0:
            left = right = int(size_diff / 2)
        else:
            left = int(size_diff / 2)
            right = size_diff - left
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=[255, 255, 255])
    else:
        size_diff = img.shape[1] - img.shape[0]
        top = 0
        bottom = 0
        left = 0
        right = 0
        if size_diff % 2 == 0:
            top = bottom = int(size_diff / 2)
        else:
            top = int(size_diff / 2)
            bottom = size_diff - top
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=[255, 255, 255])