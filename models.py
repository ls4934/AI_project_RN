import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_coordinates(batch_size):
    np_coordinates = np.zeros((batch_size, 25, 2))
    for i in range(25):
        np_coordinates[:,i,:] = np.array([i // 5, i % 5]) / 2 - 1
    return torch.FloatTensor(np_coordinates)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(256)
        
    def forward(self, image):
        x = self.conv1(image)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

class gTheta(nn.Module):
    def __init__(self):
        super(gTheta, self).__init__()
        self.g_fc1 = nn.Linear((256 + 2) * 2 + 11, 2000)
        self.g_fc2 = nn.Linear(2000, 2000)
        self.g_fc3 = nn.Linear(2000, 2000)
        self.g_fc4 = nn.Linear(2000, 2000)

    def forward(self, x):
        x = F.relu(self.g_fc1(x))
        x = F.relu(self.g_fc2(x))
        x = F.relu(self.g_fc3(x))
        x = F.relu(self.g_fc4(x))
        return x

class fPhi(nn.Module):
    def __init__(self):
        super(fPhi, self).__init__()
        self.f_fc1 = nn.Linear(2000, 2000)
        self.f_fc2 = nn.Linear(2000, 1000)
        self.f_fc3 = nn.Linear(1000, 500)
        self.f_fc4 = nn.Linear(500, 100)

    def forward(self, x):
        x = F.relu(self.f_fc1(x))
        x = F.relu(self.f_fc2(x))
        x = F.relu(self.f_fc3(x))
        x = F.relu(self.f_fc4(x))
        return x

class RN(nn.Module):
    def __init__(self):
        super(RN, self).__init__()
        self.conv = CNNModel()
        self.f_phi = fPhi()
        self.g_theta = gTheta()
        self.output_layer = nn.Linear(100, 10)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, image, questions):
        # dimension: batch_size x 256 x 5 x 5
        x = self.conv(image) 
        batch_size, channels, feat_size, _ = x.size()

        # dimension: batch_size x 25 x 256
        x = x.view(batch_size, channels, feat_size ** 2).permute(0, 2, 1)
        
        # find positions to the feature maps
        sptial_positions = get_coordinates(batch_size).to(self.device)

        # add spatial coordinates to the last dimension of the input
        x = torch.cat([x, sptial_positions], 2)               # dim: (batch_size x 25 x (256 + 2))

        # combine all object pairs with questions
        object_1 = torch.unsqueeze(x, 1)                      # dim: (batch_size x 1 x 25 x 258)
        object_1 = object_1.repeat(1, 25, 1, 1)               # dim: (batch_size x 25 x 25 x 258)
        object_2 = torch.unsqueeze(x, 2)                      # dim: (batch_size x 25 x 1 x 258)
        object_2 = object_2.repeat(1, 1, 25, 1)               # dim: (batch_size x 25 x 25 x 258)
        
        # combine questions with each pair of objects
        questions = torch.unsqueeze(questions, 1)             # dim: (batch_size x 1 x 11)
        questions = questions.repeat(1, 25, 1)                # dim: (batch_size x 25 x 11)
        questions = torch.unsqueeze(questions, 2)             # dim: (batch_size x 25 x 1 x 11)
        questions = questions.repeat(1, 1, 25, 1)             # dim: (batch_size x 25 x 25 x 11)
        
        # concatenate all together
        x = torch.cat([object_1, object_2, questions], 3)     # dim: (batch_size x 25 x 25 x (258 x 2 + 11))
        
        # reshape for passing through network
        x = x.view(batch_size * feat_size ** 4, -1)
        x = self.g_theta(x)
        
        # element-wise sum
        x = x.view(batch_size, feat_size ** 4, 2000)
        x = x.sum(1).squeeze()

        x = self.f_phi(x)
        x = F.dropout(x)
        x = F.relu(self.output_layer(x))
        return F.log_softmax(x,dim=1)

