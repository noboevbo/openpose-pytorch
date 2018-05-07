import numpy as np
import time
import cv2

from config import cfg
import concurrent.futures


def get_pad_column(img, column='left', pad_color=None):
    if column == 'left':
        column_val = img[:,:1:]
    elif column == 'right':
        column_val = img[:,-1:,:]
    elif column == 'top':
        column_val = img[:1,:,:]
    else: # bottom
        column_val = img[-1:,:,:]
    if pad_color: # else just use the values from the column
        column_val = column_val * 0 # Set color values to zero
        column_val = column_val + np.array(pad_color, dtype=np.uint8) # Fill with defined color values
    return column_val


def pad_by_stride(img, stride, pad_color):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(get_pad_column(img_padded, 'up', pad_color), (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(get_pad_column(img_padded, 'left', pad_color), (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(get_pad_column(img_padded, 'down', pad_color), (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(get_pad_column(img_padded, 'right', pad_color), (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def measure_time(method):
    def timed(*args, **kw):
        if not cfg.general.debug_timers:
            return method(*args, **kw)
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.5f sec' % (method.__name__, te-ts))
        return result

    return timed


def get_map_as_numpy(in_map):
    if cfg.network.use_gpu == 1:
        return in_map.cpu().numpy()
    else:
        return in_map.numpy()


def __get_avg_map(idx, result, original_img, maps_name, interpolation):
    start_time = time.time()
    maps = get_map_as_numpy(result[maps_name].data[0])
    #print("{}: {}".format("Get From GPU", time.time() - start_time))
    start_time = time.time()
    pad = result["pad"]
    imageToTest_padded = result["img_to_test_padded"]
    heatmap = np.transpose(maps, (1, 2, 0))
    heatmap = cv2.resize(heatmap, (0, 0), fx=cfg.general.stride, fy=cfg.general.stride,
                         interpolation=interpolation)
    heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]), interpolation=interpolation)
    heatmap = np.transpose(heatmap, (2, 0, 1))
    #print("{}: {}".format("Actual calc", time.time() - start_time))
    return idx, heatmap


def get_upsampled_maps(img_results: dict, maps_name, interpolation=cv2.INTER_CUBIC):
    original_img = img_results["original_img"]
    num_maps = img_results["results"][0][maps_name].data.shape[1]
    maps_average = np.zeros((len(img_results["results"]), num_maps, original_img.shape[0], original_img.shape[1]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(__get_avg_map, idx, result, original_img, maps_name, interpolation=interpolation)
                   for idx, result in enumerate(img_results["results"])]
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            idx, result = future.result()
            maps_average[idx] = result

    return maps_average


def get_average_map_from_upscaled_maps(img: np.ndarray):
    # This step may take a while because it needs to wait for the gpu operations to finish.
    # Could use torch.cuda.synchronize() after network output to wait for sync directly after the output
    img = np.mean(img, 0)
    img = np.transpose(img, (1, 2, 0))
    return img


def debug_additional_timer(name, start_time):
    if cfg.general.additional_debug_timers:
        print("{}: {}".format(name, time.time() - start_time))


def get_num_params(model):
    """
    Returns the number of parameters of a model
    """
    num_parameters=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        num_parameters += nn
    return num_parameters


def getEquidistantPoints(p1, p2, num_samples):
    """
    Returns num_samples points between p1 / p2 evenly distributed
    :param p1: 
    :param p2: 
    :param num_samples: 
    :return: 
    """
    return list(zip(np.linspace(p1[0], p2[0], num_samples), np.linspace(p1[1], p2[1], num_samples)))
