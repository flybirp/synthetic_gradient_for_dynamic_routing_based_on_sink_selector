import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from mlp_net import Net

input_size = 28*28
hidden_size = 500
output_size = 10
batch_size = 128

# Normalization for MNIST dataset.
dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
train_dataset = datasets.MNIST('data', train=True, download=False, transform=dataset_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

test_dataset = datasets.MNIST('data', train=False, download=False, transform=dataset_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

net = Net(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
net.load_state_dict(torch.load('dynamic_routing_net'))

correct_cnt, ave_loss = 0, 0
for batch_idx, (x, target) in enumerate(test_loader):
    x, target = Variable(x, volatile=True), Variable(target, volatile=True)
    score, loss = net(x, target)
    _, pred_label = torch.max(score.data, 1)
    correct_cnt += (pred_label == target.data).sum()
    ave_loss += loss.data[0]
accuracy = correct_cnt * 1.0 / len(test_loader) / batch_size
ave_loss /= len(test_loader)
print 'TEST>>> epoch: {}, test loss: {:.6f}, accuracy: {:.4f}'.format(1, ave_loss, accuracy)