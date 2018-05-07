import os

from configs.TrainingStateConfigHandler import TrainingStateConfigHandler

curr_dir = os.path.dirname(os.path.realpath(__file__))


class GeneralConfig(object):
    def __init__(self):
        self.input_width = 368
        self.input_height = 368
        self.stride = 8
        self.debug_timers = False
        self.additional_debug_timers = False

# Network


class NetworkConfigBase(object):
    def __init__(self, default_path):
        self.use_gpu = 1
        self.gpu_device_number = 0
        self.model_state_file = ""
        self.scale_search = [0.5, 1.0, 1.5, 2.0]
        self.pad_color = 255

        self.heatmap_thresh = 0.1

        self.limb_num_samples = 10
        self.limb_sample_score_thresh = 0.25
        self.limb_samples_over_thresh = 0.5

        self.skeleton_min_limbs = 4
        self.skeleton_limb_score = 0.25
        self.stage_delay_epochs = [0, 0, 0, 0, 0, 0] # Delays the training of a given stage to the given epoch


class NetworkConfigRtPose2D(NetworkConfigBase):
    def __init__(self, default_path):
        super().__init__(default_path)
        self.model_state_file = os.path.join(default_path, "pretrained_models", "rtpose2d", "rtpose2d.pth")
        self.training = TrainingConfigRtPose2D(default_path)


class NetworkConfigOpenPose(NetworkConfigBase):
    def __init__(self, default_path):
        super().__init__(default_path)
        self.model_state_file = os.path.join(default_path, "pretrained_models", "rtpose2d", "rtpose2d.pth")
        self.training = TrainingConfigRtPose2D(default_path)
        self.paf_thre = 8.0
        self.paf_num_samples = 10
        self.paf_thresh_sample_score = 0.05
        self.paf_samples_over_thresh = 0.8


# Training

class TrainingConfigBase(object):
    def __init__(self, default_path):
        self.name = self.__class__.__name__
        self.__checkpoint_config_handler = self.get_training_state_config_handler()
        self.checkpoint_cfg = self.__checkpoint_config_handler.config

        self.checkpoint_model_base_dir = os.path.join(default_path, "checkpoints")
        self.checkpoint_model_path = self.checkpoint_cfg[self.name]["checkpoint_model_path"]
        self.checkpoint_epoch = int(self.checkpoint_cfg[self.name]["checkpoint_epoch"])
        best_loss = self.checkpoint_cfg[self.name]["checkpoint_best_model_loss"]
        self.checkpoint_best_model_loss = float(best_loss) if best_loss else None

        self.trained_model_dir = os.path.join(default_path, "trained_models")
        self.log_dir = os.path.join(default_path, "logs")

        self.learning_rate = 0.00004  # Base learning rate, is changed per layer grp
        self.batch_size = 10
        self.gamma = 0.333  # 4 Learning Rate Scheduler
        self.stepsize = 210392  # in original code each epoch is 121746 and step change is on 17th epoch

        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.augmentation = AugmentationConfig()

    def get_training_state_config_handler(self):
        state_config_handler = TrainingStateConfigHandler()
        if self.name not in state_config_handler.config:
            state_config_handler.config[self.name] = {}
            state_config_handler.config[self.name]["checkpoint_model_path"] = ''
            state_config_handler.config[self.name]["checkpoint_epoch"] = 0
            state_config_handler.config[self.name]["checkpoint_best_model_loss"] = ''
            state_config_handler.config.write()
        return state_config_handler

    def __update_checkpoint_from_cfg(self):
        self.checkpoint_model_path = self.checkpoint_cfg[self.name]["checkpoint_model_path"]
        self.checkpoint_epoch = int(self.checkpoint_cfg[self.name]["checkpoint_epoch"])
        self.checkpoint_best_model_loss = self.checkpoint_cfg[self.name]["checkpoint_best_model_loss"]

    def update_checkpoint(self, checkpoint_model_path, checkpoint_epoch, best_model_loss):
        self.checkpoint_cfg[self.name]["checkpoint_model_path"] = checkpoint_model_path
        self.checkpoint_cfg[self.name]["checkpoint_epoch"] = checkpoint_epoch
        self.checkpoint_cfg[self.name]["checkpoint_best_model_loss"] = best_model_loss
        self.checkpoint_cfg.write()
        self.__update_checkpoint_from_cfg()


class TrainingConfigRtPose2D(TrainingConfigBase):
    def __init__(self, default_path):
        super().__init__(default_path)
        self.trained_model_dir = os.path.join(self.trained_model_dir, "rtpose2d")
        self.log_dir = os.path.join(self.log_dir, "rtpose2d")
        self.checkpoint_model_base_dir = os.path.join(self.checkpoint_model_base_dir, "rtpose2d")


class TrainingConfigOpenPose(TrainingConfigBase):
    def __init__(self, default_path):
        super().__init__(default_path)
        self.trained_model_dir = os.path.join(self.trained_model_dir, "openpose")
        self.log_dir = os.path.join(self.log_dir, "openpose")
        self.checkpoint_model_base_dir = os.path.join(self.checkpoint_model_base_dir, "openpose")


# Augmentation

class AugmentationConfig(object):
    def __init__(self):
        self.target_dist = 0.6
        self.scale_prob = 1
        self.scale_min = 0.4
        self.scale_max = 1.3
        self.max_rotate_degree = 50
        self.center_perterb_max = 50
        self.flip_prob = 0.5
        self.sigma = 7
        self.sigma_limb = 6


# Dataset

class DatasetConfigBase(object):
    def __init__(self, dataset_dir):
        self.base_dir = dataset_dir
        self.train_annotation_dir = ""
        self.train_img_dir = ""
        # This setting is used in the coco_to_hdf5 converter
        self.train_convert_hdf5 = ""
        # This setting is used for training
        self.train_hdf5 = ""

        self.val_annotation_dir = ""
        self.val_img_dir = ""
        # This setting is used in the coco_to_hdf5 converter
        self.val_convert_hdf5 = ""
        # This setting is used for validate
        self.val_hdf5 = ""


class DatasetConfigCoco(DatasetConfigBase):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.base_dir = dataset_dir
        self.train_annotation_dir = os.path.join(self.base_dir, "annotations/person_keypoints_train2017.json")
        self.train_img_dir = os.path.join(self.base_dir, "train2017")
        self.train_convert_hdf5 = os.path.join(self.base_dir, "coco_train.h5")
        self.train_hdf5 = os.path.join(self.base_dir, "train_dataset.h5")

        self.val_annotation_dir = os.path.join(self.base_dir, "annotations/person_keypoints_val2017.json")
        self.val_img_dir = os.path.join(self.base_dir, "val2017")
        self.val_convert_hdf5 = os.path.join(self.base_dir, "coco_val.h5")
        self.val_hdf5 = os.path.join(self.base_dir, "val_dataset.h5")
        self.val_size = 2637


# Convert

class ConvertConfig(object):
    def __init__(self):
        self.caffe = ConvertCaffeConfig()


class ConvertCaffeConfig(object):
    def __init__(self):
        self.caffe_model = "/home/USERNAME/git/openpose/models/pose/coco/pose_iter_440000.caffemodel"
        self.deploy_file = "/home/USERNAME/git/openpose/models/pose/coco/pose_deploy_linevec.prototxt"
        self.pytorch_model = "/media/USERNAME/Data/Dump/pose_iter_440000.pth"
        self.test_image = "/home/USERNAME/git/openpose/examples/media/COCO_val2014_000000000474.jpg"


class ConfigBase(object):
    @property
    def general(self) -> GeneralConfig:
        raise NotImplementedError

    @property
    def convert(self) -> ConvertConfig:
        raise NotImplementedError

    @property
    def network(self) -> NetworkConfigBase:
        raise NotImplementedError

    @property
    def train(self) -> TrainingConfigBase:
        raise NotImplementedError

    @property
    def dataset(self) -> DatasetConfigBase:
        raise NotImplementedError
