import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Callable, Optional
from torch import Tensor
from typing import Optional

import torch
import torch.nn as nn

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
        input_soft = torch.sigmoid(input)
        
        # compute the actual focal loss
        weight_pos = target*torch.pow(1. - input_soft, self.gamma)
        #weight_pos = target
        focal_pos = -self.alpha * weight_pos * torch.log(input_soft + self.eps)

        # compute the actual focal loss
        weight_neg = (1. - target)*torch.pow(input_soft, self.gamma)
        focal_neg = -(1. - self.alpha) * weight_neg * torch.log(1. - input_soft + self.eps)

        loss = focal_pos + focal_neg

        return torch.mean(loss, dim=-1)



class GroupedBCEWithLogitsLoss(nn.Module):
    def __init__(self,
                 pos_weight: Optional[Tensor] = None,
                 group_reduction="mean",
                 eps=1e-6
                 ) -> None:
        super(GroupedBCEWithLogitsLoss, self).__init__()
        if pos_weight is None:
            pos_weight = 1
        self.pos_weight = float(pos_weight.item())
        self.group_reduction = group_reduction


    def forward(self, input: Tensor, target: Tensor, group_id: Tensor) -> Tensor:
        unique = group_id.unique()
        loss = 0
        nu = len(unique)
        for i in range(nu):
            bool_selection = unique[i] == group_id
            reduced_input = torch.mean(input[bool_selection])
            grouped_target = target[bool_selection]
            reduced_target = torch.max(grouped_target)
            loss += self.pos_weight*reduced_target*nn.functional.logsigmoid(reduced_input)
            loss += (1.0-reduced_target)*nn.functional.logsigmoid(1-reduced_input)

        return -loss / nu


if __name__ == "__main__":
    loss = FocalLoss(alpha=1, gamma=2)
    print(loss(torch.tensor([[10.0]]), torch.tensor([[1.0]])))
    print(loss(torch.tensor([[0.0]]), torch.tensor([[1.0]])))
    print(loss(torch.tensor([[-1]]), torch.tensor([[0.0]])))