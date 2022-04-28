import torch
import torch.nn as nn


class LocalMaxGlobalMin(nn.Module):

    def __init__(self, rho, nchannels, nparts=1, device='cpu'):
        super(LocalMaxGlobalMin, self).__init__()
        self.nparts = nparts
        self.device = device
        self.nchannels = nchannels
        self.rho = rho

        nlocal_channels_norm = nchannels // self.nparts
        reminder = nchannels % self.nparts
        nlocal_channels_last = nlocal_channels_norm
        if reminder != 0:
            nlocal_channels_last = nlocal_channels_norm + reminder
        # seps records the indices partitioning feature channels into separate parts
        seps = []
        sep_node = 0
        for i in range(self.nparts):
            if i != self.nparts - 1:
                sep_node += nlocal_channels_norm
                # seps.append(sep_node)
            else:
                sep_node += nlocal_channels_last
            seps.append(sep_node)
        self.seps = seps

    def forward(self, x):
        x = x.pow(2)
        intra_x = []
        inter_x = []
        for i in range(self.nparts):
            if i == 0:
                intra_x.append((1 - x[:, :self.seps[i], :self.seps[i]]).mean())
            else:
                intra_x.append((1 - x[:, self.seps[i - 1]:self.seps[i], self.seps[i - 1]:self.seps[i]]).mean())
                inter_x.append(x[:, self.seps[i - 1]:self.seps[i], :self.seps[i - 1]].mean())

        loss = self.rho * 0.5 * (sum(intra_x) / self.nparts + sum(inter_x) / (self.nparts * (self.nparts - 1) / 2))

        return loss