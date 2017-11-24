import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from dynamic_routing_net import Net

use_cuda = torch.cuda.is_available()

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
net = net.cuda() if use_cuda else net

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)




for epoch in xrange(10):
    selector_loss = Variable(torch.FloatTensor([1]))
    # trainning
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = Variable(x), Variable(target)
        if use_cuda: x, target = x.cuda(), target.cuda()

        _, loss, synthetic_loss = net(x, target, selector_loss)

        selector_loss = Variable(torch.FloatTensor([loss.data[0]]))

        synthetic_loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

        # print net.select_path.weight[0][0]
        if batch_idx % 100 == 0:
            print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, loss.data[0])

    # testing
    correct_cnt, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        if use_cuda: x, target = x.cuda(), target.cuda()
        score, loss, synthetic_loss = net(x, target, selector_loss)
        _, pred_label = torch.max(score.data, 1)
        correct_cnt += (pred_label == target.data).sum()
        ave_loss += loss.data[0]
    accuracy = correct_cnt*1.0/len(test_loader)/batch_size
    ave_loss /= len(test_loader)
    print 'TEST>>> epoch: {}, test loss: {:.6f}, accuracy: {:.4f}'.format(epoch, ave_loss, accuracy)

torch.save(net.state_dict(), net.name())