#criterian
##RMSE LOSS
import model
import data
from data import*
from model import*
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
dataset = NumbersDataset(ligand_path, protein_path, affinity_path)
train_loader, test_loader = load_splitset(dataset, .2)


class rmsloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss()
    def forward(self,yhat,y):
       return torch.sqrt(self.mse(yhat,y))
model2 = CNNcom()

def train_d(model,train):
#criterion = nn.MSELoss()
    criterion = rmsloss()
    # model2 = CNNcom()
    optimizer=optim.Adam(model.parameters(),lr=0.003)

    num_epochs = 3
    for epoch in range(num_epochs):
       r_loss = 0
       loss1 = []
       out1 = []
       count = 0
       for i, j in (train):
           m=i[0]
           m = m.reshape((4, 62, 50))
           p=i[1]
           p = p.reshape((4, 25, 600))
           output = model(m, p)
           out1.append(output)
           loss = criterion(output, j)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           loss1.append(loss)
           count += 1
           if count == 50:
               break
    return loss1,out1


tt = train_d(model2,train_loader)
torch.save(model2,'deep.pt')
if __name__ == '__main__':
     #traind = train_d(train_loader)
     print(tt[0])
     print(tt[1])


