## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1st CNN - input image channel (grayscale 1x224x224), 32 output channels/feature maps, 
        # 3x3 square convolution kernel, stride=1
        # output_size = (W-F)/S+1 = (224-3)/1+1 = 222
        self.conv1 = nn.Conv2d(1, 32, 3)

        # maxpooling layer: kernal_size=2, stride=2
        # output_size = (222-2)/2+1 = 111
        self.pool = nn.MaxPool2d(2, 2)

        # 2nd CNN - 64 output channels/feature maps, 
        # 3x3 square convolution kernel, stride=1
        # output_size = (111-3)/1 +1 = 109
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # maxpooling layer: kernel_size=2,  stride=2
        # output_size = (109-2)/2+1 = 54 (rounded down)
        # reusing self.pool without needing to define extra

        # fully connected layer: 64x54x54 output from previous layer
        self.fc1 = nn.Linear(64*54*54, 680)

        # dropout layer with prob=0.4
        self.fc1_drop = nn.Dropout(p=0.4)

        # fully connected layer to create final 136 output 68 pair of (x, y)
        self.fc2 = nn.Linear(680, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        # x is the input image
        # two conv+relu+pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatten 
        x = x.view(x.size(0), -1)

        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # final output
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # output_size_1 = (224-3)/1+1 = 222
        # output_size_2 = (222-3)/1+1 = 220
        # output_size_2_pool = (220-2)/2+1=110
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64 , 3)  

        # output_size_3 = (110-3)/1+1 = 108
        # output_size_4 = (108-3)/1+1 = 106
        # ouput_size_4_pool = (106-2)/2+1=53
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)        

        # output_size_5 = (53-3)/1+1 = 51
        # output_size_6 = (51-3)/1+1 = 49
        # output_size_6_pool = (49-2)/2+1 = 24
        self.conv5 = nn.Conv2d(128, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)

        self.conv2_drop = nn.Dropout2d(p=0.5)                
        self.pool = nn.MaxPool2d(2, 2)        

        self.fc1 = nn.Linear(128 * 24 * 24, 1024)
        self.fc2 = nn.Linear(1024, 136)  
        

    def forward(self, x):

        x = self.conv2_drop(self.pool(F.relu(self.conv2(F.relu(self.conv1(x))))))
        x = self.conv2_drop(self.pool(F.relu(self.conv4(F.relu(self.conv3(x))))))
        x = self.conv2_drop(self.pool(F.relu(self.conv6(F.relu(self.conv5(x))))))        
        x = x.view(-1, 128 * 24 * 24)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
            
        return x
