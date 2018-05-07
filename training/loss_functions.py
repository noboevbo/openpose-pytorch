import torch
from torch.autograd import Variable


class L2Loss(torch.nn.Module):
    def __init__(self, batch_size):
        super(L2Loss, self).__init__()
        self.batch_size = batch_size

    def forward(self, x: Variable, y: Variable, weights: Variable = None):
        if weights is not None:
            val = (x-y) * weights[:x.data.shape[0], :, :, :] # Slice by shape[n,..] for batch size (last batch < batch_size)
        else:
            val = x-y
        l = torch.sum(val ** 2) / self.batch_size / 2
        return l