import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size_2 = hidden_size / 2
        self.hidden_size_3 = hidden_size / 4
        self.output_size = output_size

        self.fc0 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size_2)
        self.fc2 = nn.Linear(in_features=self.hidden_size_2, out_features=self.hidden_size_3)
        self.fc_out = nn.Linear(in_features=self.hidden_size_3, out_features=self.output_size)

        self.ceriation = nn.CrossEntropyLoss()

    def forward(self, x, target):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc_out(x))
        loss = self.ceriation(x, target)
        return x, loss

    def name(self):
        return "mlp_net"