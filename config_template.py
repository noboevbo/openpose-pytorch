from configs.Validator import ObjectValidator
from configs.base_config import GeneralConfig, ConvertConfig, ConfigBase, \
    DatasetConfigCoco, NetworkConfigOpenPose, TrainingConfigOpenPose
from configs.config_schema import cfg_schema


default_path = "/home/USERNAME/rtpose2d_data/"
dataset_dir = "/home/USERNAME/datasets/COCO"


class OpenPoseConfig(ConfigBase):
    general = GeneralConfig()
    convert = ConvertConfig()
    network = NetworkConfigOpenPose(default_path)
    train = TrainingConfigOpenPose(default_path)
    dataset = DatasetConfigCoco(dataset_dir)

    def __init__(self):
        super().__init__()
        self.network.model_state_file = "/media/disks/beta/models/openpose/itsc18_sim_full_c48.pth"

        self.train.batch_size = 10
        self.train.learning_rate = 0.001


cfg = OpenPoseConfig()

cfg_validator = ObjectValidator(cfg_schema)
if not cfg_validator.validate_object(cfg):
    raise SystemError(str(cfg_validator.errors))

