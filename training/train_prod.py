import os

import matplotlib
matplotlib.use('Agg')

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import cfg
from models.model_base import NetworkModelBase
from models.model_handler_base import NetworkModelHandlerBase
from training.loss_functions import L2Loss
from training.tensorboard_logger import TensorboardLogger, log_tensorboard_train_details, log_tensorboard_net_params, \
    log_tensorboard_map_imgs
from training.train_utils import get_per_parameter_optimizer_settings, get_learning_rate_decay_lambdas


def train(network: NetworkModelBase, data_loader_train: DataLoader, get_losses_func, loss_weights_tuple: () = None, fix_regex=None):
    logger = TensorboardLogger(cfg.train.log_dir)
    cudnn.benchmark = True

    lr_per_parameter = get_per_parameter_optimizer_settings(network.named_parameters(), fix_regex)

    optimizer = optim.SGD(lr_per_parameter,
                          lr=cfg.train.learning_rate,
                          momentum=cfg.train.momentum,
                          weight_decay=cfg.train.weight_decay)

    scheduler = LambdaLR(optimizer, lr_lambda=get_learning_rate_decay_lambdas(len(data_loader_train)))
    criterion = L2Loss(cfg.train.batch_size).cuda()

    for epoch in range(0, cfg.train.checkpoint_epoch):
        scheduler.step()

    for epoch in range(cfg.train.checkpoint_epoch, 90):
        print("Begin train epoch: {}".format(epoch))
        train_epoch(data_loader_train, network, criterion, optimizer, epoch, logger, get_losses_func, loss_weights_tuple)
        scheduler.step()
        save_checkpoint(network, epoch)


def save_checkpoint(network, epoch):
    checkpoint_model_path = os.path.join(cfg.train.checkpoint_model_base_dir, 'checkpoint_{}.pth'.format(epoch))
    torch.save(network.state_dict(), checkpoint_model_path)
    cfg.train.update_checkpoint(checkpoint_model_path, epoch, None)


def train_epoch(train_loader, network, criterion, optimizer, epoch, logger, get_losses_func, loss_weights_tuple: ()):
    num_previous_iterations = epoch * len(train_loader)
    num_samples = len(train_loader)
    percentage_0_1 = int(num_samples * 0.001)
    percentage_10 = int(num_samples * 0.1)
    percentage_25 = int(num_samples * 0.25)
    for iteration, data in enumerate(train_loader):
        print("epoch {} [{}/{}]".format(epoch, iteration, num_samples))
        image_var = Variable(data['image'].cuda())
        joint_map_gt_var = Variable(data['joint_map_gt'].cuda())
        limb_map_gt_var = Variable(data['limb_map_gt'].cuda())
        joint_map_mask_var = Variable(data['joint_map_masks'].cuda())
        limb_map_mask_var = Variable(data['limb_map_masks'].cuda())
        optimizer.zero_grad()  # zero the gradient buffer
        output = network(image_var, joint_map_mask_var, limb_map_mask_var, epoch)

        losses = get_losses_func(criterion, output, (joint_map_gt_var, limb_map_gt_var), loss_weights_tuple)

        total_loss = sum(losses)
        total_loss.backward()
        optimizer.step()

        if iteration % percentage_0_1 == 0:
            log_tensorboard_train_details(logger, num_previous_iterations + iteration, output, losses, total_loss)
        if iteration % percentage_10 == 0:
            log_tensorboard_net_params(logger, num_previous_iterations + iteration, network)
        if iteration % percentage_25 == 0:
            log_tensorboard_map_imgs(logger, num_previous_iterations + iteration, image_var, joint_map_gt_var,
                                     limb_map_gt_var,
                                     output)


# Validation

# def validate_network(network, val_loader, criterion, best_model_loss, train_iteration, logger: TensorboardLogger, get_losses_func):
#     network.eval()
#     iteration_losses = AverageMeter()
#     for iteration, data in enumerate(val_loader):
#         image_var = Variable(data['image'].cuda())
#         joint_map_gt_var = Variable(data['joint_map_gt'].cuda())
#         limb_map_gt_var = Variable(data['limb_map_gt'].cuda())
#         joint_map_mask_var = Variable(data['joint_map_masks'].cuda())
#         limb_map_mask_var = Variable(data['limb_map_masks'].cuda())
#
#         output = network(image_var, joint_map_mask_var, limb_map_mask_var)
#
#         losses = get_losses_func(criterion, output, joint_map_gt_var, limb_map_gt_var)
#
#         total_loss = sum(losses)
#         iteration_losses.update(total_loss.data[0], image_var.size(0))
#
#     if best_model_loss:
#         is_best = iteration_losses.avg < best_model_loss
#         best_model = min(best_model_loss, iteration_losses.avg)
#     else:
#         is_best = True
#         best_model = iteration_losses.avg
#     logger.scalar_summary('val_loss', iteration_losses.avg, train_iteration)
#
#     network.train()
#     return is_best, best_model


def get_network_model(network_model_handler: NetworkModelHandlerBase):
    network_model = network_model_handler.get_train_model()
    if cfg.train.checkpoint_epoch == 0:  # Fresh training
        print("Load pretrained feature extractor weights")
        network_model = network_model_handler.load_pretrained_feature_extractor_parameters(network_model)
    else:
        print("Load from checkpoint: {}".format(cfg.train.checkpoint_model_path))
        network_model = torch.load(cfg.train.checkpoint_model_path)
    return network_model
