import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
#import torchnet
#from torchnet import meter
from sklearn.metrics import confusion_matrix
import time
import itertools
import sys

# Hyper Parameters
input_size = 3 * 96 * 96
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 4
learning_rate = 0.001

start = time.time()
#%%--------------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#%%--------------------------------------------------------------------------------------------

train_set = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print('\nThe training data set has %d samples' % len(train_set))


#%%--------------------------------------------------------------------------------------------

loadEndTime = time.time()
print("Time to load: ", loadEndTime - start)

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



# MultiLayer Perceptron Neural Network

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

# --------------------------------------------------------------------------------------------

net = Net(input_size, hidden_size, num_classes)
print(net)
net.cuda()

# --------------------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# --------------------------------------------------------------------------------------------

# Train the Model
trainStart = time.time()


for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        # Convert torch tensor to Variable
        images, labels = data
        images = images.view(-1, input_size)

        # wrap them in Variable
        images, labels = Variable(images.cuda()), Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 #       confusion_matrix.add(outputs.data, labels.data)
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.data[0]))
print('Finished Training')
print('time to train', time.time() - trainStart)
#print(confusion_matrix.conf)


# Test the Model
print('\nThe test data set has %d samples' % len(testset))

correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, input_size).cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 50000 test images: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------

_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))
# --------------------------------------------------------------------------------------------


class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
for data in test_loader:
    images, labels = data
    images = Variable(images.view(-1, input_size).cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.cuda()).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(num_classes):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# --------------------------------------------------------------------------------------------
print ("Total time", time.time() - start)