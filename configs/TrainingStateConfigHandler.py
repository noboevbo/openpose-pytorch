import os

from configobj import ConfigObj

curr_dir = os.path.dirname(os.path.realpath(__file__))


class TrainingStateConfigHandler:
    def __init__(self, config_path=os.path.join(curr_dir, "training_state_config.ini")):
        self.config_path = config_path
        self.config = None
        self.load_config()

    def load_config(self):
        self.config = ConfigObj(self.config_path)