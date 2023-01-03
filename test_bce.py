import torch.nn as nn
import torch


def true_bce(input, target, pos_weight=None):
    max_val = (-input).clamp_min_(0)
    if (pos_weight is not None):
        log_weight = (pos_weight - 1).mul(target).add_(1);
        loss = (1 - target).mul_(input).add_(log_weight.mul_(((-max_val).exp_().add_((-input - max_val).exp_())).log_().add_(max_val)))
    else:
        loss = (1 - target).mul_(input).add_(max_val).add_((-max_val).exp_().add_((-input -max_val).exp_()).log_())
    #loss = ((1-target)*input + max_val) + torch.log( torch.exp(-max_val)+torch.exp(-input -max_val) )

    return torch.mean(loss)

def my_bce(input, target, pos_weight=None):
    if pos_weight is not None:
        return -torch.mean(pos_weight*target*nn.functional.logsigmoid(input) + (1-target)*nn.functional.logsigmoid(1-input) )
    return -torch.mean(target*nn.functional.logsigmoid(input) + (1-target)*nn.functional.logsigmoid(1-input) )
    #return -torch.mean(target*log_sigmoid(input) + (1-target)*log_sigmoid(1-input) )


y_true = torch.tensor([0.0, 0.0, 1.40, 0.3, 1.0, 0.0, 0.3, 0.0])
y_out = torch.tensor([ 0.3, 0.2, 0.0, -2 , -4 , 3.0, 2, 0.0])

loss1 = nn.functional.binary_cross_entropy_with_logits(y_out, y_true)
loss2 = my_bce(y_out, y_true)
print(loss1.item(), loss2.item())

y_true = torch.tensor([0.0, 0.0, 1.0, 0.0])
y_out =  torch.tensor([-100.0, -100, 100, -100])

loss1 = nn.functional.binary_cross_entropy_with_logits(y_out, y_true)
loss2 = my_bce(y_out, y_true)

print(loss1.item(), loss2.item())

###################################

y_true = torch.tensor([0.0, 0.0, 1.40, 0.3, 1.0, 0.0, 0.3, 0.0])
y_out = torch.tensor([ 0.3, 0.2, 0.0, -2 , -4 , 3.0, 2, 0.0])

loss1 = nn.functional.binary_cross_entropy_with_logits(y_out, y_true, pos_weight=torch.tensor([204]))
loss2 = my_bce(y_out, y_true, pos_weight=204)
print(loss1.item(), loss2.item())
