import cv2
import numpy as np
import torch
from torch.autograd import Variable

from utils import util
from config import cfg
from models.model_handler_base import NetworkModelBase
from util_img import normalize


@util.measure_time
def get_pytorch_from_numpy(img):
    pytorch_img = np.transpose(img, (2, 0, 1))  # transpose to channels, height, width
    pytorch_img = np.expand_dims(pytorch_img, axis=0)  # add dim
    pytorch_img = normalize(pytorch_img)
    pytorch_img = torch.from_numpy(pytorch_img).float()
    if cfg.network.use_gpu == 1:
        pytorch_img = pytorch_img.pin_memory().cuda()
    return Variable(pytorch_img)


@util.measure_time
def get_network_results(model: NetworkModelBase, original_img):
    scales = cfg.network.scale_search
    multiplier_width = [x * cfg.general.input_width / original_img.shape[1] for x in scales]
    img_result = {
        "original_img": original_img,
        "results": []
    }

    for idx, m in enumerate(range(len(scales))):
        scale_x = multiplier_width[m]
        imageToTest = cv2.resize(original_img, (0, 0), fx=scale_x, fy=scale_x, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.pad_by_stride(imageToTest, cfg.general.stride, cfg.network.pad_color)
        pytorch_img = get_pytorch_from_numpy(imageToTest_padded)
        with torch.no_grad():
            limb_maps_output, joint_maps_output = model(pytorch_img)

        img_result["results"].append({
            "img_to_test_padded": imageToTest_padded,
            "pad": pad,
            "joint_maps": joint_maps_output,
            "limb_maps": limb_maps_output
        })

    return img_result
