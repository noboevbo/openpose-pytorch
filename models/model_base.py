from torch import nn

from config import cfg


class NetworkModelBase(nn.Module):
    @property
    def num_joint_maps(self) -> int:
        raise NotImplementedError

    @property
    def num_limb_maps(self) -> int:
        raise NotImplementedError

    @property
    def num_layers(self) -> int:
        return self.num_joint_maps + self.num_limb_maps

    @property
    def limb_maps_start(self) -> int:
        return 0

    @property
    def joint_maps_start(self) -> int:
        return self.num_limb_maps

    @property
    def limb_map_bg(self) -> int:
        return self.num_limb_maps - 1

    @property
    def joint_map_bg(self) -> int:
        return self.num_layers - 1

    @property
    def parts_shape(self) -> (int, int, int):
        return self.num_layers, cfg.general.input_height // cfg.general.stride, cfg.general.input_width // cfg.general.stride

    @property
    def mask_shape(self) -> (int, int):
        return cfg.general.input_height // cfg.general.stride, cfg.general.input_width // cfg.general.stride  # 46, 46

    @property
    def data_shape(self) -> (int, int, int):
        return 3, cfg.general.input_height, cfg.general.input_width
