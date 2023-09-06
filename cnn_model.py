import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNNCIFAR10(nn.Module):
    # CNN model for the CIFAR-10 dataset

    def __init__(
        self,
        structure= [64, 'M', 128, 'M', 256, 'M', 512,'M', 10,'M']
    ): 
        super(ConvNNCIFAR10, self).__init__()
        self.structure=structure
        self.arc=self.make()

    def make(self,batch_norm=True):
        layers = []
        in_channels = 3
        structure=self.structure
        for v in structure:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True),nn.Dropout(0.1)]
                else:
                    layers += [conv2d, nn.ReLU(True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.arc(x)
        x = torch.flatten(x, 1) 
        # The embedding space, it is possible to use other layers as the embedding space
        e = x.clone().detach()
        self.e=e
        
        return x


class ConvNNMNIST(nn.Module):
    # CNN model for the MNIST dataset
    
    def __init__(
        self
    ):
        super(ConvNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 10, 2)
        self.b1=nn.BatchNorm2d(64)
        self.b2=nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.b1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 4)
        x = self.conv2(x)
        x = self.b2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = F.max_pool2d(x, 4)
        x = self.conv3(x)
       
        x = torch.flatten(x, 1)
        e = x.clone().detach()
        self.e=e
        
        return x




    




