from collections import OrderedDict

import torch
from torch import nn

from utils import util
from config import cfg
from models.model_base import NetworkModelBase


def load_parameters_by_layer_structure(network: nn.Module, model_state_dict: OrderedDict):
    pretrained_model_keys = list(model_state_dict.keys())
    net_keys = list(network.state_dict().keys())
    weights_load = {}
    for i in range(len(net_keys)):
        weights_load[net_keys[i]] = model_state_dict[pretrained_model_keys[i]]
    state = network.state_dict()
    state.update(weights_load)
    network.load_state_dict(state)
    return network


class NetworkModelHandlerBase(object):
    @util.measure_time
    def get_model(self) -> NetworkModelBase:
        raise NotImplementedError

    @util.measure_time
    def get_train_model(self) -> NetworkModelBase:
        raise NotImplementedError

    def load_state_dict(self, network: nn.Module, state_dict_path: str = cfg.network.model_state_file):
        state_dict = torch.load(state_dict_path)
        network.load_state_dict(state_dict)

    def load_pretrained_feature_extractor_parameters(self, network:nn.Module):
        raise NotImplementedError

    def load_pretrained_stage1_parameters(self, network: nn.Module):
        raise NotImplementedError
