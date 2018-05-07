import time
from collections import OrderedDict

import torch.nn.functional as F
from torch import nn, cat
from torchvision import models

from utils import util

from config import cfg
from models.model_handler_base import NetworkModelHandlerBase, NetworkModelBase


class OpenPoseModelHandler(NetworkModelHandlerBase):
    @util.measure_time
    def get_model(self):
        network = OpenPoseModel()
        if cfg.network.use_gpu == 1:
            network.float()
            network.cuda()
        return network

    @util.measure_time
    def get_train_model(self):
        network = OpenPoseTrainModel()
        if cfg.network.use_gpu == 1:
            network.float()
            network.cuda()
        return network

    def load_pretrained_feature_extractor_parameters(self, network: nn.Module):
        vgg19 = models.vgg19(pretrained=True)
        vgg19_state_dict = vgg19.state_dict()
        vgg19_keys = list(vgg19_state_dict.keys())
        net_keys = list(network.state_dict().keys())
        weights_load = {}
        for i in range(20):
            weights_load[net_keys[i]] = vgg19_state_dict[vgg19_keys[i]]
        state = network.state_dict()
        state.update(weights_load)
        network.load_state_dict(state)
        return network

    def load_pretrained_stage1_parameters(self, network: nn.Module):
        # todo
        # openpose_state_dict = openpose.state_dict()
        # openpose_keys = list(openpose_state_dict.keys())
        # net_keys = list(network.state_dict().keys())
        # weights_load = {}
        # for i in range(20):
        #     weights_load[net_keys[i]] = openpose_state_dict[openpose_keys[i]]
        # state = network.state_dict()
        # state.update(weights_load)
        # network.load_state_dict(state)
        # return network
        raise NotImplementedError


class OpenPoseModel(NetworkModelBase):
    num_joint_maps = 19
    num_limb_maps = 38
    num_limb_in = 128 + num_limb_maps
    num_limb_joint_in = 128 + num_limb_maps + num_joint_maps
    limb_paf_mapping = [[12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3],
                        [4, 5], [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35],
                        [32, 33], [36, 37], [18, 19], [26, 27]]

    def __init__(self):
        super(OpenPoseModel, self).__init__()
        # Feature Extractor
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4_3_CPM = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                                     padding=1)
        self.conv4_4_CPM = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Stage 1

        self.stage1_1_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                                            padding=1)
        self.stage1_2_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                                            padding=1)
        self.stage1_3_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                                            padding=1)
        self.stage1_4_limb_maps = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1),
                                            padding=0)
        self.stage1_5_limb_maps = nn.Conv2d(in_channels=512, out_channels=self.num_limb_maps, kernel_size=(1, 1), stride=(1, 1),
                                            padding=0)

        self.stage1_1_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                                             padding=1)
        self.stage1_2_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                                             padding=1)
        self.stage1_3_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                                             padding=1)
        self.stage1_4_joint_maps = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1),
                                             padding=0)
        self.stage1_5_joint_maps = nn.Conv2d(in_channels=512, out_channels=self.num_joint_maps, kernel_size=(1, 1), stride=(1, 1),
                                             padding=0)

        # Stage 2

        self.stage2_1_limb_maps = nn.Conv2d(in_channels=self.num_limb_joint_in, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage2_2_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage2_3_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage2_4_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage2_5_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage2_6_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        self.stage2_7_limb_maps = nn.Conv2d(in_channels=128, out_channels=self.num_limb_maps, kernel_size=(1, 1), stride=1, padding=0)

        self.stage2_1_joint_maps = nn.Conv2d(in_channels=self.num_limb_joint_in, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage2_2_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage2_3_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage2_4_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage2_5_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage2_6_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        self.stage2_7_joint_maps = nn.Conv2d(in_channels=128, out_channels=self.num_joint_maps, kernel_size=(1, 1), stride=1, padding=0)

        # Stage 3

        self.stage3_1_limb_maps = nn.Conv2d(in_channels=self.num_limb_joint_in, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage3_2_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage3_3_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage3_4_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage3_5_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage3_6_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        self.stage3_7_limb_maps = nn.Conv2d(in_channels=128, out_channels=self.num_limb_maps, kernel_size=(1, 1), stride=1, padding=0)

        self.stage3_1_joint_maps = nn.Conv2d(in_channels=self.num_limb_joint_in, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage3_2_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage3_3_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage3_4_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage3_5_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage3_6_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        self.stage3_7_joint_maps = nn.Conv2d(in_channels=128, out_channels=self.num_joint_maps, kernel_size=(1, 1), stride=1, padding=0)

        # Stage 4

        self.stage4_1_limb_maps = nn.Conv2d(in_channels=self.num_limb_joint_in, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage4_2_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage4_3_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage4_4_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage4_5_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage4_6_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        self.stage4_7_limb_maps = nn.Conv2d(in_channels=128, out_channels=self.num_limb_maps, kernel_size=(1, 1), stride=1, padding=0)

        self.stage4_1_joint_maps = nn.Conv2d(in_channels=self.num_limb_joint_in, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage4_2_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage4_3_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage4_4_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage4_5_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage4_6_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        self.stage4_7_joint_maps = nn.Conv2d(in_channels=128, out_channels=self.num_joint_maps, kernel_size=(1, 1), stride=1, padding=0)

        # Stage 5

        self.stage5_1_limb_maps = nn.Conv2d(in_channels=self.num_limb_joint_in, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage5_2_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage5_3_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage5_4_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage5_5_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage5_6_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        self.stage5_7_limb_maps = nn.Conv2d(in_channels=128, out_channels=self.num_limb_maps, kernel_size=(1, 1), stride=1, padding=0)

        self.stage5_1_joint_maps = nn.Conv2d(in_channels=self.num_limb_joint_in, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage5_2_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage5_3_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage5_4_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage5_5_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage5_6_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        self.stage5_7_joint_maps = nn.Conv2d(in_channels=128, out_channels=self.num_joint_maps, kernel_size=(1, 1), stride=1, padding=0)

        # Stage 6

        self.stage6_1_limb_maps = nn.Conv2d(in_channels=self.num_limb_joint_in, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage6_2_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage6_3_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage6_4_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage6_5_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage6_6_limb_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        self.stage6_7_limb_maps = nn.Conv2d(in_channels=128, out_channels=self.num_limb_maps, kernel_size=(1, 1), stride=1, padding=0)

        self.stage6_1_joint_maps = nn.Conv2d(in_channels=self.num_limb_joint_in, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage6_2_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage6_3_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage6_4_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage6_5_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding=3)
        self.stage6_6_joint_maps = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1, padding=0)
        self.stage6_7_joint_maps = nn.Conv2d(in_channels=128, out_channels=self.num_joint_maps, kernel_size=(1, 1), stride=1, padding=0)

        self.out_feature = None

        self.out_stage1_limb_maps = None
        self.out_stage1_joint_maps = None

        self.out_stage2_limb_maps = None
        self.out_stage2_joint_maps = None

        self.out_stage3_limb_maps = None
        self.out_stage3_joint_maps = None

        self.out_stage4_limb_maps = None
        self.out_stage4_joint_maps = None

        self.out_stage5_limb_maps = None
        self.out_stage5_joint_maps = None

        self.out_stage6_limb_maps = None
        self.out_stage6_joint_maps = None

    def forward_feature_extraction(self, x):
        out = F.relu(self.conv1_1(x), inplace=True)
        out = F.relu(self.conv1_2(out), inplace=True)
        out = self.pool1_stage1(out)
        out = F.relu(self.conv2_1(out), inplace=True)
        out = F.relu(self.conv2_2(out), inplace=True)
        out = self.pool2_stage1(out)
        out = F.relu(self.conv3_1(out), inplace=True)
        out = F.relu(self.conv3_2(out), inplace=True)
        out = F.relu(self.conv3_3(out), inplace=True)
        out = F.relu(self.conv3_4(out), inplace=True)
        out = self.pool3_stage1(out)
        out = F.relu(self.conv4_1(out), inplace=True)
        out = F.relu(self.conv4_2(out), inplace=True)
        out = F.relu(self.conv4_3_CPM(out), inplace=True)
        return F.relu(self.conv4_4_CPM(out), inplace=True)

    def forward_stage_1_limb_maps(self, input_var):
        out = F.relu(self.stage1_1_limb_maps(input_var), inplace=True)
        out = F.relu(self.stage1_2_limb_maps(out), inplace=True)
        out = F.relu(self.stage1_3_limb_maps(out), inplace=True)
        out = F.relu(self.stage1_4_limb_maps(out), inplace=True)
        return self.stage1_5_limb_maps(out)

    def forward_stage_1_joint_maps(self, input_var):
        out = F.relu(self.stage1_1_joint_maps(input_var), inplace=True)
        out = F.relu(self.stage1_2_joint_maps(out), inplace=True)
        out = F.relu(self.stage1_3_joint_maps(out), inplace=True)
        out = F.relu(self.stage1_4_joint_maps(out), inplace=True)
        return self.stage1_5_joint_maps(out)

    def forward_stage2_limb_maps(self, input_var):
        out = F.relu(self.stage2_1_limb_maps(input_var), inplace=True)
        out = F.relu(self.stage2_2_limb_maps(out), inplace=True)
        out = F.relu(self.stage2_3_limb_maps(out), inplace=True)
        out = F.relu(self.stage2_4_limb_maps(out), inplace=True)
        out = F.relu(self.stage2_5_limb_maps(out), inplace=True)
        out = F.relu(self.stage2_6_limb_maps(out), inplace=True)
        return self.stage2_7_limb_maps(out)

    def forward_stage2_joint_maps(self, input_var):
        out = F.relu(self.stage2_1_joint_maps(input_var), inplace=True)
        out = F.relu(self.stage2_2_joint_maps(out), inplace=True)
        out = F.relu(self.stage2_3_joint_maps(out), inplace=True)
        out = F.relu(self.stage2_4_joint_maps(out), inplace=True)
        out = F.relu(self.stage2_5_joint_maps(out), inplace=True)
        out = F.relu(self.stage2_6_joint_maps(out), inplace=True)
        return self.stage2_7_joint_maps(out)

    def forward_stage3_limb_maps(self, input_var):
        out = F.relu(self.stage3_1_limb_maps(input_var), inplace=True)
        out = F.relu(self.stage3_2_limb_maps(out), inplace=True)
        out = F.relu(self.stage3_3_limb_maps(out), inplace=True)
        out = F.relu(self.stage3_4_limb_maps(out), inplace=True)
        out = F.relu(self.stage3_5_limb_maps(out), inplace=True)
        out = F.relu(self.stage3_6_limb_maps(out), inplace=True)
        return self.stage3_7_limb_maps(out)

    def forward_stage3_joint_maps(self, input_var):
        out = F.relu(self.stage3_1_joint_maps(input_var), inplace=True)
        out = F.relu(self.stage3_2_joint_maps(out), inplace=True)
        out = F.relu(self.stage3_3_joint_maps(out), inplace=True)
        out = F.relu(self.stage3_4_joint_maps(out), inplace=True)
        out = F.relu(self.stage3_5_joint_maps(out), inplace=True)
        out = F.relu(self.stage3_6_joint_maps(out), inplace=True)
        return self.stage3_7_joint_maps(out)

    def forward_stage4_limb_maps(self, input_var):
        out = F.relu(self.stage4_1_limb_maps(input_var), inplace=True)
        out = F.relu(self.stage4_2_limb_maps(out), inplace=True)
        out = F.relu(self.stage4_3_limb_maps(out), inplace=True)
        out = F.relu(self.stage4_4_limb_maps(out), inplace=True)
        out = F.relu(self.stage4_5_limb_maps(out), inplace=True)
        out = F.relu(self.stage4_6_limb_maps(out), inplace=True)
        return self.stage4_7_limb_maps(out)

    def forward_stage4_joint_maps(self, input_var):
        out = F.relu(self.stage4_1_joint_maps(input_var), inplace=True)
        out = F.relu(self.stage4_2_joint_maps(out), inplace=True)
        out = F.relu(self.stage4_3_joint_maps(out), inplace=True)
        out = F.relu(self.stage4_4_joint_maps(out), inplace=True)
        out = F.relu(self.stage4_5_joint_maps(out), inplace=True)
        out = F.relu(self.stage4_6_joint_maps(out), inplace=True)
        return self.stage4_7_joint_maps(out)

    def forward_stage5_limb_maps(self, input_var):
        out = F.relu(self.stage5_1_limb_maps(input_var), inplace=True)
        out = F.relu(self.stage5_2_limb_maps(out), inplace=True)
        out = F.relu(self.stage5_3_limb_maps(out), inplace=True)
        out = F.relu(self.stage5_4_limb_maps(out), inplace=True)
        out = F.relu(self.stage5_5_limb_maps(out), inplace=True)
        out = F.relu(self.stage5_6_limb_maps(out), inplace=True)
        return self.stage5_7_limb_maps(out)

    def forward_stage5_joint_maps(self, input_var):
        out = F.relu(self.stage5_1_joint_maps(input_var), inplace=True)
        out = F.relu(self.stage5_2_joint_maps(out), inplace=True)
        out = F.relu(self.stage5_3_joint_maps(out), inplace=True)
        out = F.relu(self.stage5_4_joint_maps(out), inplace=True)
        out = F.relu(self.stage5_5_joint_maps(out), inplace=True)
        out = F.relu(self.stage5_6_joint_maps(out), inplace=True)
        return self.stage5_7_joint_maps(out)

    def forward_stage6_limb_maps(self, input_var):
        out = F.relu(self.stage6_1_limb_maps(input_var), inplace=True)
        out = F.relu(self.stage6_2_limb_maps(out), inplace=True)
        out = F.relu(self.stage6_3_limb_maps(out), inplace=True)
        out = F.relu(self.stage6_4_limb_maps(out), inplace=True)
        out = F.relu(self.stage6_5_limb_maps(out), inplace=True)
        out = F.relu(self.stage6_6_limb_maps(out), inplace=True)
        return self.stage6_7_limb_maps(out)

    def forward_stage6_joint_maps(self, input_var):
        out = F.relu(self.stage6_1_joint_maps(input_var), inplace=True)
        out = F.relu(self.stage6_2_joint_maps(out), inplace=True)
        out = F.relu(self.stage6_3_joint_maps(out), inplace=True)
        out = F.relu(self.stage6_4_joint_maps(out), inplace=True)
        out = F.relu(self.stage6_5_joint_maps(out), inplace=True)
        out = F.relu(self.stage6_6_joint_maps(out), inplace=True)
        return self.stage6_7_joint_maps(out)

    @util.measure_time
    def forward(self, x):
        # Feature Extraction
        self.out_feature = self.forward_feature_extraction(x)
        start_time = time.time()

        # Stage 1
        self.out_stage1_limb_maps = self.forward_stage_1_limb_maps(self.out_feature)
        self.out_stage1_joint_maps = self.forward_stage_1_joint_maps(self.out_feature)

        # Stage 2

        concat_stage2 = cat([self.out_stage1_limb_maps, self.out_stage1_joint_maps, self.out_feature], 1)
        self.out_stage2_limb_maps = self.forward_stage2_limb_maps(concat_stage2)
        self.out_stage2_joint_maps = self.forward_stage2_joint_maps(concat_stage2)

        # Stage 3

        concat_stage3 = cat([self.out_stage2_limb_maps, self.out_stage2_joint_maps, self.out_feature], 1)
        self.out_stage3_limb_maps = self.forward_stage3_limb_maps(concat_stage3)
        self.out_stage3_joint_maps = self.forward_stage3_joint_maps(concat_stage3)

        # Stage 4

        concat_stage4 = cat([self.out_stage3_limb_maps, self.out_stage3_joint_maps, self.out_feature], 1)
        self.out_stage4_limb_maps = self.forward_stage4_limb_maps(concat_stage4)
        self.out_stage4_joint_maps = self.forward_stage4_joint_maps(concat_stage4)

        # Stage 5

        concat_stage5 = cat([self.out_stage4_limb_maps, self.out_stage4_joint_maps, self.out_feature], 1)
        self.out_stage5_limb_maps = self.forward_stage5_limb_maps(concat_stage5)
        self.out_stage5_joint_maps = self.forward_stage5_joint_maps(concat_stage5)

        # Stage 6

        concat_stage6 = cat([self.out_stage5_limb_maps, self.out_stage5_joint_maps, self.out_feature], 1)
        self.out_stage6_limb_maps = self.forward_stage6_limb_maps(concat_stage6)
        self.out_stage6_joint_maps = self.forward_stage6_joint_maps(concat_stage6)

        util.debug_additional_timer("Stages", start_time)
        return self.out_stage6_limb_maps, self.out_stage6_joint_maps
        # # self.output
        # concat_stage7 = cat([self.out_stage6_limb_maps, self.out_stage6_joint_maps], 1)
        #
        # return concat_stage7


class OpenPoseTrainModel(OpenPoseModel):
    def __init__(self):
        super(OpenPoseTrainModel, self).__init__()

    def forward(self, x, joint_map_masks, limb_map_masks, epoch):
        """
        Applies the ground truth miss masks to the output of each stage because the ground truth is also masked!
        """
        result_dict = OrderedDict()
        # Feature Extraction
        self.out_feature = self.forward_feature_extraction(x)

        # Stage 1
        self.out_stage1_limb_maps = self.forward_stage_1_limb_maps(self.out_feature)
        self.out_stage1_joint_maps = self.forward_stage_1_joint_maps(self.out_feature)
        result_dict[1] = OrderedDict({
            "limb_map": self.out_stage1_limb_maps * limb_map_masks,
            "joint_map": self.out_stage1_joint_maps * joint_map_masks,
        })

        # Stage 2

        if epoch >= cfg.network.stage_delay_epochs[0]:
            concat_stage2 = cat([self.out_stage1_limb_maps, self.out_stage1_joint_maps, self.out_feature], 1)
            self.out_stage2_limb_maps = self.forward_stage2_limb_maps(concat_stage2)
            self.out_stage2_joint_maps = self.forward_stage2_joint_maps(concat_stage2)
            result_dict[2] = OrderedDict({
                "limb_map": self.out_stage2_limb_maps * limb_map_masks,
                "joint_map": self.out_stage2_joint_maps * joint_map_masks,
            })

        # Stage 3

        if epoch >= cfg.network.stage_delay_epochs[1]:
            concat_stage3 = cat([self.out_stage2_limb_maps, self.out_stage2_joint_maps, self.out_feature], 1)
            self.out_stage3_limb_maps = self.forward_stage3_limb_maps(concat_stage3)
            self.out_stage3_joint_maps = self.forward_stage3_joint_maps(concat_stage3)
            result_dict[3] = OrderedDict({
                "limb_map": self.out_stage3_limb_maps * limb_map_masks,
                "joint_map": self.out_stage3_joint_maps * joint_map_masks,
            })

        # Stage 4
        if epoch >= cfg.network.stage_delay_epochs[2]:
            concat_stage4 = cat([self.out_stage3_limb_maps, self.out_stage3_joint_maps, self.out_feature], 1)
            self.out_stage4_limb_maps = self.forward_stage4_limb_maps(concat_stage4)
            self.out_stage4_joint_maps = self.forward_stage4_joint_maps(concat_stage4)
            result_dict[4] = OrderedDict({
                "limb_map": self.out_stage4_limb_maps * limb_map_masks,
                "joint_map": self.out_stage4_joint_maps * joint_map_masks,
            })
        # Stage 5

        if epoch >= cfg.network.stage_delay_epochs[3]:
            concat_stage5 = cat([self.out_stage4_limb_maps, self.out_stage4_joint_maps, self.out_feature], 1)
            self.out_stage5_limb_maps = self.forward_stage5_limb_maps(concat_stage5)
            self.out_stage5_joint_maps = self.forward_stage5_joint_maps(concat_stage5)
            result_dict[5] = OrderedDict({
                "limb_map": self.out_stage5_limb_maps * limb_map_masks,
                "joint_map": self.out_stage5_joint_maps * joint_map_masks,
            })

        # Stage 6

        if epoch >= cfg.network.stage_delay_epochs[4]:
            concat_stage6 = cat([self.out_stage5_limb_maps, self.out_stage5_joint_maps, self.out_feature], 1)
            self.out_stage6_limb_maps = self.forward_stage6_limb_maps(concat_stage6)
            self.out_stage6_joint_maps = self.forward_stage6_joint_maps(concat_stage6)
            result_dict[6] = OrderedDict({
                "limb_map": self.out_stage6_limb_maps * limb_map_masks,
                "joint_map": self.out_stage6_joint_maps * joint_map_masks,
            })

        return result_dict
