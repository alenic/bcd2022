import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Callable, Optional
from torch import Tensor
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: Optional[float] = 2.0) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
            ) -> torch.Tensor:

        # compute softmax over the classes axis
        input_soft = F.sigmoid(input)
        
        # compute the actual focal loss
        weight_pos = target*torch.pow(1. - input_soft, self.gamma)
        focal_pos = -self.alpha * weight_pos * torch.log(input_soft + self.eps)

        # compute the actual focal loss
        weight_neg = (1. - target)*torch.pow(input_soft, self.gamma)
        focal_neg = -(1. - self.alpha) * weight_neg * torch.log(1. - input_soft + self.eps)

        loss = focal_pos + focal_neg

        return torch.mean(loss)


if __name__ == "__main__":
    loss = FocalLoss(alpha=0.8, gamma=2)
    print(loss(torch.tensor([[1.0]]), torch.tensor([[1.0]])))