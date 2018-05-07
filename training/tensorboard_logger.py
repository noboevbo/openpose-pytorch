# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from collections import OrderedDict

import tensorflow as tf
import numpy as np
import cv2
import torch

from util_img import get_img_from_network_output
from visualization import get_network_train_output

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class TensorboardLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, image_dict, step):
        """Log a list of images."""

        img_summaries = []
        for i, (key, img) in enumerate(image_dict.items()):
            # Write the image to a string
            s = cv2.imencode(".png", img)[1].tostring()

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s,
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s_%s' % (tag, key), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)


    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def log_tensorboard_train_details(logger: TensorboardLogger, iteration_total, outputs: OrderedDict, losses, total_loss):
    info = {'1_loss_total': total_loss.data[0]}
    count = 0
    for stage_nr, stage_layers in outputs.items():
        for name, _ in stage_layers.items():
            loss_name = "loss_s{}_{}".format(stage_nr, name)
            info[loss_name] = losses[count]
            count += 1

    for tag, value in info.items():
        logger.scalar_summary(tag, value, iteration_total)


def log_tensorboard_net_params(logger: TensorboardLogger, iteration_total, net: torch.nn.Module):
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), iteration_total)
        if value.grad is not None:
            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), iteration_total)


def log_tensorboard_map_imgs(logger: TensorboardLogger, iteration_total, img_var, joint_maps_gt_var, limb_maps_gt_var,
                             outputs: OrderedDict):
    # TODO: Add a summary img (grid with img, limb maps)
    img_dict = {'original': get_img_from_network_output(img_var[0].data.cpu().numpy())}
    for stage_nr, stage_layers in outputs.items():
        limb_maps = None
        joint_maps = None
        for name, value in stage_layers.items():
            if name == "limb_map":
                limb_maps = value[0]
            if name == "joint_map":
                joint_maps = value[0]
        img = get_network_train_output(joint_maps_gt_var[0].data.cpu().numpy(), limb_maps_gt_var[0].data.cpu().numpy(),
                                       joint_maps.data.cpu().numpy(), limb_maps.data.cpu().numpy(),
                                       get_img_from_network_output(img_var[0].data.cpu().numpy()))
        img_dict['stage{}'.format(stage_nr)] = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    logger.image_summary('maps', img_dict, iteration_total)

