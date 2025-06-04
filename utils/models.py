import torch.nn as nn
from utils.utilities import DEBUG

class MyNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MyNet, self).__init__()

        if "n_classes" in kwargs:
            n_classes = kwargs["n_classes"]
        else:
            print("n_classes arg not found, using 10 by default")
            n_classes = 10

        self.conv1 = nn.Conv2d(3,16,(3,3), stride=(1,1), padding='valid')
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1))
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16,8,(3,3), stride=(1,1), padding='valid')
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1))
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(8,1,(3,3), stride=(1,1), padding='valid')
        self.pool3 = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1))
        self.relu3 = nn.ReLU()

        self.flat = nn.Flatten()

        self.dense1 = nn.Linear(400, 512)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.dense2 = nn.Linear(512, n_classes)

    
    def residual(self, x):
        pass
    
    def forward(self, x):
        if DEBUG: print(f"input shape: {x.shape}")

        # Block 1
        x = self.conv1(x)
        x= self.bn1(x)
        if DEBUG: print(f"conv1 shape: {x.shape}")
        x = self.pool1(x)
        if DEBUG: print(f"pool1 shape: {x.shape}")
        x = self.relu1(x)
        if DEBUG: print(f"relu1 shape: {x.shape}")

        # Block 2
        x = self.conv2(x)
        x= self.bn2(x)
        if DEBUG: print(f"conv2 shape: {x.shape}")
        x = self.pool2(x)
        if DEBUG: print(f"pool2 shape: {x.shape}")
        x = self.relu2(x)
        if DEBUG: print(f"relu2 shape: {x.shape}")

        # Block 3
        x = self.conv3(x)
        if DEBUG: print(f"conv3 shape: {x.shape}")
        x = self.pool3(x)
        if DEBUG: print(f"pool3 shape: {x.shape}")
        x = self.relu3(x)
        if DEBUG: print(f"relu3 shape: {x.shape}")

        x = self.flat(x)
        if DEBUG: print(f"flat shape: {x.shape}")

        x = self.dense1(x)
        x = self.dropout1(x)
        if DEBUG: print(f"dense1 shape: {x.shape}")

        x = self.relu4(x)
        x = self.dense2(x)
        if DEBUG: print(f"dense2 shape: {x.shape}")

        return x