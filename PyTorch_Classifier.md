
# Set Training and Test data


```python
import torch
import torchvision
import torchvision.transforms as transforms
import time
```


```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

testset
```

    Files already downloaded and verified
    Files already downloaded and verified





    Dataset CIFAR10
        Number of datapoints: 10000
        Root location: ./data
        Split: Test



# Define a CNN


```python
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 3 input channel, 6 output channel, 5x5 conv kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 2x2 maxpool
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# intantiate
net = Net()
```

# Choose loss and optimizer function


```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

```

# Training on CPU


```python
device = torch.device("cpu")
net.to(device)
```




    Net(
      (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=400, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )




```python
t = time.time()
# epoch = 5
for epoch in range(5): 

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data[0].to(device) , data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # optimize
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # every 2000 mini-batches
            print('[%d, %5d] loss: %.2f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            # initialize to zero for next epoch
            running_loss = 0.0

print('Finished!')

print('Time taken on CPU:',time.time() - t)
```

    [1,  2000] loss: 2.21
    [1,  4000] loss: 1.94
    [1,  6000] loss: 1.72
    [1,  8000] loss: 1.60
    [1, 10000] loss: 1.57
    [1, 12000] loss: 1.49
    [2,  2000] loss: 1.42
    [2,  4000] loss: 1.40
    [2,  6000] loss: 1.35
    [2,  8000] loss: 1.34
    [2, 10000] loss: 1.35
    [2, 12000] loss: 1.31
    [3,  2000] loss: 1.22
    [3,  4000] loss: 1.25
    [3,  6000] loss: 1.23
    [3,  8000] loss: 1.21
    [3, 10000] loss: 1.20
    [3, 12000] loss: 1.18
    [4,  2000] loss: 1.12
    [4,  4000] loss: 1.11
    [4,  6000] loss: 1.13
    [4,  8000] loss: 1.12
    [4, 10000] loss: 1.11
    [4, 12000] loss: 1.12
    [5,  2000] loss: 1.04
    [5,  4000] loss: 1.01
    [5,  6000] loss: 1.03
    [5,  8000] loss: 1.04
    [5, 10000] loss: 1.05
    [5, 12000] loss: 1.07
    Finished!
    Time taken on CPU: 270.08389806747437


# Training On GPU


```python
net = Net()
# one GPU
device = torch.device('cuda:0')
print(device)
net.to(device)
```

    cuda:0





    Net(
      (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=400, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )




```python
t = time.time()
# epoch = 5
for epoch in range(5): 

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data[0].to(device) , data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # optimize
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # every 2000 mini-batches
            print('[%d, %5d] loss: %.2f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            # initialize to zero for next epoch
            running_loss = 0.0

print('Finished!')

print('Time taken on GPU:',time.time() - t)
```

    [1,  2000] loss: 2.30
    [1,  4000] loss: 2.30
    [1,  6000] loss: 2.30
    [1,  8000] loss: 2.30
    [1, 10000] loss: 2.31
    [1, 12000] loss: 2.31
    [2,  2000] loss: 2.30
    [2,  4000] loss: 2.31
    [2,  6000] loss: 2.31
    [2,  8000] loss: 2.30
    [2, 10000] loss: 2.30
    [2, 12000] loss: 2.30
    [3,  2000] loss: 2.30
    [3,  4000] loss: 2.30
    [3,  6000] loss: 2.31
    [3,  8000] loss: 2.30
    [3, 10000] loss: 2.30
    [3, 12000] loss: 2.30
    [4,  2000] loss: 2.31
    [4,  4000] loss: 2.30
    [4,  6000] loss: 2.31
    [4,  8000] loss: 2.31
    [4, 10000] loss: 2.30
    [4, 12000] loss: 2.30
    [5,  2000] loss: 2.31
    [5,  4000] loss: 2.30
    [5,  6000] loss: 2.30
    [5,  8000] loss: 2.30
    [5, 10000] loss: 2.31
    [5, 12000] loss: 2.30
    Finished!
    Time taken on GPU: 174.4679172039032


### Training took ~ 270 secs on CPU and ~174 secs on GTX 1050 GPU


```python

```
