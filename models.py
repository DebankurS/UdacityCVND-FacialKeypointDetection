## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Input dimension (1,224,224)
        self.conv1 = nn.Conv2d(1, 32, 5) # Output (32,220,220)  (W-F)/S+1=(224-5)/1+1=220
        # After Pool (32,110,110)
        self.conv2 = nn.Conv2d(32,64,3) # Output (64,108,108)  (W-F)/S+1=(110-3)/1+1=108
        # After Pool (32,54,54) 
        self.conv3 = nn.Conv2d(64,128, 3) # Output (128,52,52)  (W-F)/S+1=(54-3)/1+1=52
        # After Pool (128,26,26)
        self.pool = nn.MaxPool2d(2,2)

        self.fc1= nn.Linear(128*26*26, 1028)
        self.fc2= nn.Linear(1028, 512)
        self.fc3= nn.Linear(512, 136)

        self.dropout=nn.Dropout(0.25)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128*26*26)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
