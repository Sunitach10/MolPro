###MODEL
import data
from data import*

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
class CNNcom(nn.Module):
    def __init__(self):
        super(CNNcom, self).__init__()
        # for smiles
        self.sconv1 = nn.Conv1d(in_channels=62, out_channels=16, kernel_size=2, stride=1, padding=1)
        self.sconv2 = nn.Conv1d(16, 32, 2, stride=1, padding=1)

        self.pool = nn.MaxPool1d(2, 2)

        #for proteins
        self.pconv1 = nn.Conv1d(in_channels=25, out_channels=16, kernel_size=2, stride=1, padding=1)
        self.pconv2 = nn.Conv1d(16, 32, 2, stride=1, padding=1)


        self.dropout = nn.Dropout(0.15)
        self.linear1 = nn.Linear(5216, 256)  # put the z
        self.linear2 = nn.Linear(256, 1)

    def forward(self, x, x1):
        x = self.pool(F.relu(self.sconv1(x)))
        x = self.pool(F.relu(self.sconv2(x)))


        x1 = self.pool(F.relu(self.pconv1(x1)))
        x1 = self.pool(F.relu(self.pconv2(x1)))

        #x=x.reshape(25,32)
        #x1=x1.reshape(250,32)
        # reshape
        x = x.view(-1, 13*32)
        x1 = x1.view(-1,150*32)

        x2 = torch.cat([x, x1],1)
        x2 = F.relu(self.linear1(x2))
        x2 = self.dropout(x2)
        x2 = self.linear2(x2)
        return x2




# if __name__ == '__main__':
#     print(model2)