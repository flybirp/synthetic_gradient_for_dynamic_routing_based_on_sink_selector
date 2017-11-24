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

        self.select_path = nn.Linear(in_features=self.hidden_size, out_features=3)

        self.synthetic_gradient_layer = nn.Linear(in_features=3, out_features=1)
        self.synthetic_gradient_layer_o = nn.Linear(in_features=128, out_features=1)


        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size_2)
        self.fc11 = nn.Linear(in_features=self.hidden_size_2, out_features=self.hidden_size_3)

        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size_2)
        self.fc22 = nn.Linear(in_features=self.hidden_size_2, out_features=self.hidden_size_3)

        self.fc3 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size_2)
        self.fc33 = nn.Linear(in_features=self.hidden_size_2, out_features=self.hidden_size_3)


        self.fc_out = nn.Linear(in_features=self.hidden_size_3, out_features=self.output_size)

        self.ceriation = nn.CrossEntropyLoss()

        self.synthetic_criterion = nn.MSELoss()


    def forward(self, x, target, selector_loss):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc0(x))


        #path selector part
        selector = F.softmax(self.select_path(x))
        synthetic = F.sigmoid(self.synthetic_gradient_layer(selector))
        synthetic = torch.squeeze(synthetic, 1)
        synthetic = F.sigmoid(self.synthetic_gradient_layer_o(synthetic))
        synthetic_loss = self.synthetic_criterion(synthetic, selector_loss)

        a, index = selector.max(1)

        arr = []
        for select, x_i in enumerate(x, 0):

            i = index[select].data[0]
            if i==0:
                x_i = F.relu(self.fc1(x_i))
                x_i = F.relu(self.fc11(x_i))
            if i==1:
                x_i = F.relu(self.fc2(x_i))
                x_i = F.relu(self.fc22(x_i))
            if i==2:
                x_i = F.relu(self.fc3(x_i))
                x_i = F.relu(self.fc33(x_i))
            arr.append(x_i)

        x = torch.stack(arr)


        x = F.relu(self.fc_out(x))

        loss = self.ceriation(x, target)

        return x, loss, synthetic_loss

    def name(self):
        return "dynamic_routing_net"