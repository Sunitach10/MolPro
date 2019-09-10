import torch
import torch.nn as nn
import data_w
from data_w import*
from torch import optim
import torch.nn.functional as F
class WideCNN(nn.Module):
     def __init__(self):
         super().__init__()
         ###protein
         self.pconv1=nn.Conv1d(in_channels=6729, out_channels=16, kernel_size=2, stride=1, padding=1)
         self.pconv2=nn.Conv1d(16,32,2,stride=1, padding=1)
         self.maxpool=nn.MaxPool1d(2,2)
         ###ligands
         self.lconv1=nn.Conv1d(in_channels=10,out_channels=16,kernel_size=2,stride=1,padding=1)
         self.lconv2=nn.Conv1d(16,32,2,stride=1,padding=1)
         #####motif
         self.mconv1=nn.Conv1d(in_channels=1076,out_channels=16,kernel_size=2,stride=1,padding=1)
         self.mconv2=nn.Conv1d(16,32,2,stride=1,padding=1)

         ###
         self.dropout=nn.Dropout(.3)
         self.FC1=nn.Linear(5120,512)
         self.FC2=nn.Linear(512,10)
         self.FC3=nn.Linear(10,1)
     def forward(self,x1,x2,x3):
         x1=self.maxpool(F.relu(self.pconv1(x1)))
         x1=self.maxpool(F.relu(self.pconv2(x1)))

         x2=self.maxpool(F.relu(self.lconv1(x2)))
         x2=self.maxpool(F.relu(self.lconv2(x2)))

         x3=self.maxpool(F.relu(self.mconv1(x3)))
         x3=self.maxpool(F.relu(self.mconv2(x3)))



         x1=x1.view(-1,149*32)
         x2=x2.view(-1,3*32)
         x3=x3.view(-1,8*32)


         x=torch.cat([x1,x2,x3],1)
         x = F.relu(self.FC1(x))
         x = self.dropout(x)
         x = F.relu(self.FC2(x))
         x = self.dropout(x)
         x = self.FC3(x)
         return x

modelw=WideCNN()
modelw
if __name__ == '__main__':
    print(modelw)
























