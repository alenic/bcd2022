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