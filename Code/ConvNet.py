import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time
import torch.nn.functional as F

# Hyper Parameters
input_size = 3 *96 * 96
num_classes = 10
num_epochs = 10
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------
#%% STL10 Dataset

train_set = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#%% Show Some Test Images

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

#%% Build The Net

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=2, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=1, dilation=1, groups=1, bias=True)               
        self.fc1 = nn.Linear(96*96*16,1000)
        self.fc2 = nn.Linear(1000,100)
        self.fc3 = nn.Linear(100,10)
        

    def forward(self, input):
        x = F.relu(self.pool(self.conv1(input)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(4,-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.2)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
net = Net()
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# %%--------------------------------------------------------------------------------------------
# Train the Model
t0 = time.time()
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        # Convert torch tensor to Variable
        images, labels = data

        # wrap them in Variable
        images, labels = Variable(images.cuda()), Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.data[0]))
                 
        
print('This Job took ',time.time()-t0,' seconds to train on GPU')

# %%--------------------------------------------------------------------------------------------
# Test the Model

correct = 0
total = 0
for images, labels in train_loader:
    images = Variable(images.cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicted.cpu()==labels).sum()
    
print('Accuracy of the network on the train images: %d %%' %(100*correct/total))
# %%--------------------------------------------------------------------------------------------
# Test the Model

correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicted.cpu()==labels).sum()
    
print('Accuracy of the network on the test images: %d %%' %(100*correct/total))
#%%
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in train_loader:
    images, labels = data
    images = Variable(images.cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted.cpu() == labels).squeeze()
    for i in range(1):
       label = labels[i]
       class_correct[label] += c[i]
       class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# %%--------------------------------------------------------------------------------------------
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in test_loader:
    images, labels = data
    images = Variable(images.cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted.cpu() == labels).squeeze()
    for i in range(1):
       label = labels[i]
       class_correct[label] += c[i]
       class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

#%%

confusion = np.zeros((len(classes),len(classes)))

for data in test_loader:
    images, labels = data
    images = Variable(images.cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    for i in range(0,len( predicted)):  
        confusion[labels[i]][predicted[i]]+=1
print(confusion)
