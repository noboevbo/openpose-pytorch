import re
from collections import OrderedDict

import torch
import math

from torch.autograd import Variable
from torch.utils.data import DataLoader

from config import cfg
from skeletons.skeleton_config_base import SkeletonConfigBase
import numpy as np

layer_weight_suffix = ".weight"
layer_bias_suffix = ".bias"


def get_parameter_groups(named_parameters, fix_regex):
    """
    When using MultiGPU (DataParallel) the parameters are named module.[...] that
    :param named_parameters:
    :return:
    """
    parameter_groups = {"feature_extractor_weights": [], "feature_extractor_bias": [],
                        "stage1_weights": [], "stage1_bias": [],
                        "stageN_weights": [], "stageN_bias": []}
    for parameter in named_parameters:
        layer_name = parameter[0]
        if fix_regex and not re.match(fix_regex, layer_name):
            continue
        # Parameters - Stage 1
        if re.match("(module.)?stage1_\d_.*", layer_name):
            if layer_name.endswith(layer_weight_suffix):
                parameter_groups["stage1_weights"].append(parameter[1])
            else:
                parameter_groups["stage1_bias"].append(parameter[1])
        # Parameters - Stage 2+
        elif re.match("(module.)?stage[2-9][0-9]*_\d_.*", layer_name):
            if layer_name.endswith(layer_weight_suffix):
                parameter_groups["stageN_weights"].append(parameter[1])
            else:
                parameter_groups["stageN_bias"].append(parameter[1])
        # Parameters - Feature Extractor
        else:
            if layer_name.endswith(layer_weight_suffix):
                parameter_groups["feature_extractor_weights"].append(parameter[1])
            else:
                parameter_groups["feature_extractor_bias"].append(parameter[1])
    return parameter_groups


def get_per_parameter_optimizer_settings(named_parameters, fix_regex=None):
    parameter_groups = get_parameter_groups(named_parameters, fix_regex)
    return [{'params': parameter_groups["feature_extractor_weights"], 'lr': cfg.train.learning_rate * 1., 'weight_decay': cfg.train.weight_decay},
            {'params': parameter_groups["feature_extractor_bias"], 'lr': cfg.train.learning_rate * 2., 'weight_decay': 0},
            {'params': parameter_groups["stage1_weights"], 'lr': cfg.train.learning_rate * 1., 'weight_decay': cfg.train.weight_decay},
            {'params': parameter_groups["stage1_bias"], 'lr': cfg.train.learning_rate * 2., 'weight_decay': 0},
            {'params': parameter_groups["stageN_weights"], 'lr': cfg.train.learning_rate * 4., 'weight_decay': cfg.train.weight_decay},
            {'params': parameter_groups["stageN_bias"], 'lr': cfg.train.learning_rate * 8., 'weight_decay': 0},
            ]


def learning_rate_step_decay(epoch, iterations_per_epoch, base_learning_rate):
    steps = epoch * iterations_per_epoch * cfg.train.batch_size
    # TODO: Stepsize from cfg -> Calculate live
    lrate = base_learning_rate * math.pow(cfg.train.gamma, math.floor(steps/cfg.train.stepsize))
    print("Epoch:", epoch, "Learning rate:", lrate)
    return lrate


def get_learning_rate_decay_lambdas(num_training_samples):
    ipe = iterations_per_epoch = num_training_samples // cfg.train.batch_size
    """
    Returns a learning rate decay function for each parameter group (get_per_parameter_optimizer_settings)
    """
    return [
        lambda epoch: learning_rate_step_decay(epoch, ipe, cfg.train.learning_rate * 1.), # FeatureExtrac.Weight
        lambda epoch: learning_rate_step_decay(epoch, ipe, cfg.train.learning_rate * 2.), # FeatureExtrac.Bias
        lambda epoch: learning_rate_step_decay(epoch, ipe, cfg.train.learning_rate * 1.), # Stage1.Weight
        lambda epoch: learning_rate_step_decay(epoch, ipe, cfg.train.learning_rate * 2.), # Stage1.Bias
        lambda epoch: learning_rate_step_decay(epoch, ipe, cfg.train.learning_rate * 4.), # StageN.Weight
        lambda epoch: learning_rate_step_decay(epoch, ipe, cfg.train.learning_rate * 8.), # StageN.Bias
    ]


def fix_layers_weights(network: torch.nn.Module, fix_layer_regex):
    named_params = list(network.named_parameters())
    count = len(named_params)
    for param in named_params:
        layer_name = param[0]
        if not re.match(fix_layer_regex, layer_name):
            value = param[1]
            value.requires_grad = False
            count -= 1
    print("Params with grad: {}".format(count))


def get_loss_weights(data_loader: DataLoader, skeleton_config: SkeletonConfigBase):
    sample = next(iter(data_loader))

    sample_limb_gt = sample['limb_map_gt'].numpy()
    loss_weights_limbs = np.ones_like(sample_limb_gt)
    for limb in skeleton_config.important_limbs:
        loss_weights_limbs[:, limb, :, :] = loss_weights_limbs[:, limb, :, :] * 2

    sample_joint_gt = sample['joint_map_gt'].numpy()
    loss_weights_joints = np.ones_like(sample_joint_gt)
    for joint in skeleton_config.important_joints:
        loss_weights_joints[:, joint, :, :] = loss_weights_joints[:, joint, :, :] * 2

    return Variable(torch.from_numpy(loss_weights_limbs).cuda()), Variable(torch.from_numpy(loss_weights_joints).cuda())


def get_losses(criterion, output: OrderedDict, ground_truth_tuple: (), loss_weight_tuple: () = None) -> []:
    """
    Creates a loss for each output and maps it to the corresponding joint map / limb map.
    Iterates in steps of two to set the criterion for the joint map / limb map for each output stage
    """
    joint_map_gt_var = ground_truth_tuple[0]
    limb_map_gt_var = ground_truth_tuple[1]
    loss_weights_limbs, loss_weights_joints = None, None
    if loss_weight_tuple:
        loss_weights_limbs, loss_weights_joints = loss_weight_tuple
    losses = []
    for stage, stage_layers in output.items():
        for layer_name, layer_value in stage_layers.items():
            gt = joint_map_gt_var if layer_name == "joint_map" else limb_map_gt_var
            weight = loss_weights_joints if layer_name == "joint_map" else loss_weights_limbs
            losses.append(criterion(layer_value, gt, weight))
    return losses