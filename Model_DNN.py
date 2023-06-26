import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self,in_dim,hidden_layers:list,out_dim,dropout=0.3):
        super(DNN, self).__init__()

        self.hidden_layer_nums = len(hidden_layers)
        self.layers = []
        for i in range(self.hidden_layer_nums):
            if i == 0:
                layer = nn.Sequential(
                    nn.Linear(in_dim,hidden_layers[i]),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_layers[i])
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(hidden_layers[i-1],hidden_layers[i]),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_layers[i])
                )
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)
        self.layer_last = nn.Sequential(
            nn.Linear(hidden_layers[i],out_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

    def forward(self, x):
        for i in range(self.hidden_layer_nums):
            x = F.relu(self.layers[i](x))
        x = self.layer_last(x)
        return x