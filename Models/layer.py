from cv2 import mean
from sympy import print_rcode
import torch
import torch.nn as nn

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, out, others = ctx.saved_tensors
        gama = others[0].item()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        return grad_output * tmp, None

class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class IF(nn.Module):
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0):
        super(IF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        self.T = T
        self.loss = 0
        self.register_buffer('spike_count', torch.zeros(1))
        self.register_buffer('time_count',  torch.zeros(1))
    def forward(self, x):
        if self.T > 0:
            thre = self.thresh.data
            x = self.expand(x)
            mem = 0.5 * thre
            spike_pot = []
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = self.act(mem - thre, self.gama) * thre
                self.spike_count += spike.detach().sum()
                self.time_count  += spike.numel()
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        else:
            # stash pre-quantized potential
            self.last_mem = x.detach().clone()
            # proceed with QCFS-style quantization
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x * self.L + 0.5) / self.L
            x = x * self.thresh
        return x
    
    def reset_stats(self):
        self.spike_count.zero_()
        self.time_count.zero_()

    def get_spike_rate(self):
        # fraction of “fired” over total neuron‐time
        if self.time_count.item() > 0:
            return (self.spike_count / self.time_count).item()
        else:
            return 0.0


def add_dimention(x, T):
    x = x.unsqueeze(1)
    return x.repeat(T, 1, 1, 1, 1)
