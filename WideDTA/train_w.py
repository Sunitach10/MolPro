import data_w
from data_w import*
import model_w
from model_w import*

class rmsloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse=nn.MSELoss()
    def forward(self,yhat,y):
       return torch.sqrt(self.mse(yhat,y))

def train_w(train):
    epochs=3
    criterion = rmsloss()
    optimizer=optim.Adam(modelw.parameters(),lr=0.003)
    for epoch in range(epochs):
        outw = []
        losw = []
        k=0
        for i,j in train:
           m = i[0]
           m = m.reshape((1, 10, 10))
           p = i[1]
           p = p.reshape((1, 6729, 594))
           mt = i[2]
           mt = mt.reshape((1, 1076, 32))
           out = modelw(p, m, mt)
           outw.append(out)
           optimizer.zero_grad()
           loss = criterion(out, j)
           losw.append(loss)
           loss.backward()
           optimizer.step()
           k += 1
           if k == 50:
               break
    return outw,losw
trainwide=train_w(train_loader)
trainwide

if __name__ == '__main__':
   print(trainwide[0])
   print(trainwide[1])








