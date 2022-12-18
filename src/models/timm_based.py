import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class Timm(nn.Module):
    def __init__(self, model_name, num_classes, hidden=[], drop_rate=0):
        super().__init__()
        self.back = timm.create_model(model_name, num_classes=0, in_chans=1, pretrained=True)
        input_size = timm.get_pretrained_cfg_value(model_name, "input_size")

        print("num_classes: ", num_classes)
        print("hidden: ", hidden)
        print("drop_rate: ", drop_rate)

        with torch.no_grad():
            self.back.eval()
            n_features = self.back(torch.rand(1, 1, input_size[1], input_size[2])).shape[1]

        self.head = []
        hidden = [n_features] + hidden + [num_classes]
        for i in range(len(hidden)-1):
            linear = nn.Linear(hidden[i], hidden[i+1])
            nn.init.xavier_normal_(linear.weight)
            self.head += [nn.Dropout(drop_rate), linear, nn.ReLU(inplace=True)]
        
        self.head.pop(-1) # remove last relu
        self.head = nn.Sequential(*self.head)
    

    def forward(self, x):
        x = self.back(x)
        x = self.head(x)
        return x


class MultiHeadTimm(nn.Module):
    def __init__(self, model_name: str,
                       in_chans: int = 1,
                       hidden: int = None,
                       heads_num: list = [1],
                       inference: bool = False,
                       drop_rate: float=0.0):
        super().__init__()
        self.back = timm.create_model(model_name, num_classes=0, in_chans=in_chans, pretrained=True)
        input_size = timm.get_pretrained_cfg_value(model_name, "input_size")

        print("hidden: ", hidden)
        print("heads_num: ", heads_num)
        print("drop_rate: ", drop_rate)
        self.drop_rate = drop_rate
        self.heads_num = heads_num

        with torch.no_grad():
            self.back.eval()
            n_features = self.back(torch.rand(1, 1, input_size[1], input_size[2])).shape[1]
        
        if hidden is not None:
            self.hidden = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(inplace=True))
            n_features = hidden
        else:
            self.hidden = nn.Identity()

        self.inference = inference
        # Heads
        if inference:
            self.head = nn.Linear(n_features, 1)
            self.drop_rate = 0
        else:
            self.head = nn.ModuleList([nn.Linear(n_features, hn) for hn in heads_num])
    

    def forward(self, x):
        x = self.back(x)

        if self.training and self.drop_rate>0:
            x = F.dropout(x, p=self.drop_rate, inplace=True)
        
        x = self.hidden(x)

        if self.training and self.drop_rate>0:
            x = F.dropout(x, p=self.drop_rate, inplace=True)

        if self.inference:
            return self.head(x)
        else:
            logits = []
            for i in range(len(self.head)):
                logits.append(self.head[i](x))

            return logits



if __name__ == "__main__":
    m = MultiHeadTimm("resnet50", in_chans=1, hidden=None, heads_num=[1], inference=False)
    m.eval()
    with torch.no_grad():
        print(m(torch.rand(2, 1, 224, 224)))


    m = MultiHeadTimm("resnet50", in_chans=1, hidden=None, heads_num=[1, 2, 1], inference=False)
    m.eval()
    with torch.no_grad():
        print(m(torch.rand(2, 1, 224, 224)))


    m = MultiHeadTimm("resnet50", in_chans=1, hidden=128, heads_num=[1, 2, 1], inference=False)
    m.eval()
    with torch.no_grad():
        print(m(torch.rand(2, 1, 224, 224)))
    