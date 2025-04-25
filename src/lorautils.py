import torch
import torch.nn  as nn

class LinearWithLoRa(nn.Module):
    def __init__(self, linear, rank=16, alpha=1.5):
        super().__init__()

        self.linear = linear
        din, dout = self.linear.in_features, self.linear.out_features
        self.loralayer = LoRALayer(din=din, dout=dout, rank=rank, alpha=alpha)

    def forward(self, x):
        return self.linear(x) + self.loralayer(x)


class LoRALayer(nn.Module):
    def __init__(self,  din = 768 ,dout = 768 ,rank = 16 , alpha  = 1.5):
        super().__init__()
        self.alpha = alpha
        self.A = nn.Parameter(data = torch.empty(size =  (din ,rank ) ) )
        nn.init.kaiming_uniform_(self.A , a = 5**.5)
        self.B = nn.Parameter(data = torch.zeros((rank , dout)))
    def forward(self, x):
        return self.alpha * (x@self.A@self.B)


def ReplaceLinear(model:nn.Module , rank = 16 , alpha = 1.5):
    for name,module in model.named_children():
        if isinstance(module , nn.Linear):
            setattr(model ,name , LinearWithLoRa(module ,rank = rank , alpha=alpha))
        else :
            ReplaceLinear(model= module , rank=rank , alpha= alpha)