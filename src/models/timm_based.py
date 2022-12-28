import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from ..surgery import *

class TimmBackbone(nn.Module):
    # You must provide self.n_features
    def __init__(self, model_name: str,
                       in_chans: int = 1,
                       n_hidden: int = None,
                       drop_rate_back: float=0.0,
                       pretrained=False):
        super().__init__()
        self.back = timm.create_model(model_name, num_classes=0, in_chans=in_chans, pretrained=pretrained)
        input_size = timm.get_pretrained_cfg_value(model_name, "input_size")
        self.n_hidden = n_hidden
        self.drop_rate = drop_rate_back

        print("TimmBackbone hidden: ", n_hidden)
        print("TimmBackbone in_chans: ", in_chans)
        
        if n_hidden is not None:
            print("TimmBackbone drop_rate: ", drop_rate_back)
            self.drop_rate = drop_rate_back

        with torch.no_grad():
            self.back.eval()
            n_features = self.back(torch.rand(1, in_chans, input_size[1], input_size[2])).shape[1]
            self.n_features = n_features
        
        if n_hidden is not None:
            assert n_hidden > 0
            self.hidden_linear = nn.Sequential(nn.Linear(n_features, n_hidden), nn.ReLU())
            self.n_features = n_hidden

    
    def forward(self, x):
        x = self.back(x)

        if self.n_hidden is not None:
            if self.training and self.drop_rate>0:
                x = F.dropout(x, p=self.drop_rate)
            
            x = self.hidden_linear(x)
        
        return x


class TimmBackboneLowFeatures(nn.Module):
    # You must provide self.n_features
    def __init__(self, model_name: str,
                       in_chans: int = 1,
                       n_hidden: int = None,
                       drop_rate_back: float=0.0,
                       pretrained=False):
        super().__init__()
        self.back = timm.create_model(model_name, num_classes=0, in_chans=in_chans, pretrained=pretrained)
        input_size = timm.get_pretrained_cfg_value(model_name, "input_size")

        self.monitor = ForwardMonitor(self.back, verbose=True)
        self.layer_name = "blocks.1.3.drop_path"
        self.monitor.add_layer(self.layer_name)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(1)
        

        self.n_hidden = n_hidden
        self.drop_rate = drop_rate_back

        print("TimmBackbone hidden: ", n_hidden)
        print("TimmBackbone in_chans: ", in_chans)
        
        if n_hidden is not None:
            print("TimmBackbone drop_rate: ", drop_rate_back)
            self.drop_rate = drop_rate_back

        with torch.no_grad():
            self.back.eval()
            n_features = self.back(torch.rand(1, in_chans, input_size[1], input_size[2])).shape[1]
            n_features += self.monitor.get_layer(self.layer_name).shape[1]
            self.n_features = n_features
        
        if n_hidden is not None:
            assert n_hidden > 0
            self.hidden_linear = nn.Sequential(nn.Linear(n_features, n_hidden), nn.ReLU())
            self.n_features = n_hidden



    def forward(self, x):
        x = self.back(x)
        monitored = self.monitor.get_layer(self.layer_name)
        avg_monitored =  self.avg_pool4(monitored).squeeze(-1).squeeze(-1)
        x = torch.cat((x, avg_monitored), dim = -1)
        if self.n_hidden is not None:
            if self.training and self.drop_rate>0:
                x = F.dropout(x, p=self.drop_rate)
            
            x = self.hidden_linear(x)
        
        return x


class MultiHead(nn.Module):
    def __init__(self, backbone,
                       heads_num: list = [1],
                       inference: bool = False,
                       drop_rate_mh: float=0.0):
        super().__init__()

        print("MultiHead heads_num: ", heads_num)
        print("MultiHead inference: ", inference)
        print("MultiHead drop_rate: ", drop_rate_mh)

        self.backbone = backbone
        self.heads_num = heads_num
        self.inference = inference
        self.drop_rate = drop_rate_mh
        
        n_features = self.backbone.n_features

        # Heads
        if inference:
            self.head = nn.Linear(n_features, 1)
            self.drop_rate = 0
        else:
            self.head = nn.ModuleList([nn.Linear(n_features, hn) for hn in heads_num])
        
    def forward(self, x):
        x = self.backbone(x)

        # Dropout can be also applied for every heads
        if self.training and self.drop_rate>0:
            x = F.dropout(x, p=self.drop_rate)

        if self.inference:
            return self.head(x)
        else:
            logits = []
            for i in range(len(self.head)):
                logits.append(self.head[i](x))

            return logits


if __name__ == "__main__":
    backbone = TimmBackbone("resnet50", in_chans=1, n_hidden=None)
    m = MultiHead(backbone, heads_num=[1], inference=False)
    m.eval()
    with torch.no_grad():
        print(m(torch.rand(2, 1, 224, 224)))


    backbone = TimmBackbone("resnet50", in_chans=1, n_hidden=None)
    m = MultiHead(backbone, heads_num=[1,2,1], inference=False)
    m.eval()
    with torch.no_grad():
        print(m(torch.rand(2, 1, 224, 224)))


    backbone = TimmBackbone("resnet50", in_chans=1, n_hidden=128)
    m = MultiHead(backbone, heads_num=[1,2,1], inference=False)
    m.eval()
    with torch.no_grad():
        print(m(torch.rand(2, 1, 224, 224)))
    