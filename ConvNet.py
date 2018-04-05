#%% --------------------------------------------------------------------------------------------
# Hyper Parameters
input_size = 3 *92 * 92
num_classes = 10
num_epochs = 2
batch_size = 4
learning_rate = 0.001
#%%--------------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#%%--------------------------------------------------------------------------------------------

train_set = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%--------------------------------------------------------------------------------------------
