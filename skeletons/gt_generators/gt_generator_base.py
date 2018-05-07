class GroundTruthGeneratorBase:
    def get_ground_truth(self, joints, mask_miss):
        raise NotImplementedError