import torch
from torch.utils.data import DataLoader

from config import cfg
from datasets.coco_dataset import CocoDataset
from models import model_openpose
from skeletons.gt_generators.gt_generator_openpose import GroundTruthGeneratorOpenPose
from skeletons.skeleton_config_openpose import SkeletonConfigOpenPose
from training.train_prod import train
from training.train_utils import get_losses, fix_layers_weights

network_model_handler = model_openpose.OpenPoseModelHandler()
network = network_model_handler.get_train_model()
network_model_handler.load_state_dict(network)
fix_layers_weights(network, "stage[2-6]_[1-9]_(joint|limb)_maps")
skeleton_config = SkeletonConfigOpenPose()
gt_generator = GroundTruthGeneratorOpenPose(network, skeleton_config)


train_dataset = CocoDataset([cfg.dataset.train_hdf5], skeleton_config, gt_generator,
                            network, augment=True)
sim_dataset = CocoDataset(["/media/USERNAME/Store1/sim_train_18_04_17_ITSC.h5"], skeleton_config, gt_generator,
                          network, augment=True)
train_sets = torch.utils.data.ConcatDataset([train_dataset, sim_dataset])
train_loader = DataLoader(train_sets, cfg.train.batch_size, shuffle=True)

train(network, train_loader, get_losses, fix_regex="stage[2-6]_[1-9]_(joint|limb)_maps")
