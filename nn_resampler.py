import torch.nn as nn
import torch.nn.functional as F
import torch

class nn_resampler(nn.Module):
    
    def __init__(self, n_input, n_output):
        super(nn_resampler, self).__init__()
        
        # encoder
        self.enc1 = nn.Linear(n_input,50)
        self.enc2 = nn.Linear(50,25)
        self.enc3 = nn.Linear(25,10)
        
        # decoder
        self.dec1 = nn.Linear(10, 25)
        self.dec2 = nn.Linear(25, 50)
        self.dec3 = nn.Linear(50, n_output)
        
        #bn
        self.bn_enc1 = nn.BatchNorm1d(50)
        self.bn_enc2 = nn.BatchNorm1d(25)
        self.bn_enc3 = nn.BatchNorm1d(10)
        self.bn_dec1 = nn.BatchNorm1d(25)
        self.bn_dec2 = nn.BatchNorm1d(50)


    def forward(self, x):
        x = self.bn_enc1(torch.relu(self.enc1(x)))
        x = self.bn_enc2(torch.relu(self.enc2(x)))
        x = self.bn_enc3(torch.relu(self.enc3(x)))
        x = self.bn_dec1(torch.relu(self.dec1(x)))
        x = self.bn_dec2(torch.relu(self.dec2(x)))
        x = F.softmax(self.dec3(x), dim=1)
        return x