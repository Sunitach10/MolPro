import data
from data import*
import model
from model import*
import train
from train import*
def predict(test):
    out2 = []
    count = 0
    for i,j in(test):
        m1 = i[0]
        m1 = m1.reshape((1, 62, 50))
        p1 = i[1]
        p1 = p1.reshape((1, 25, 600))
        predict = model2(m1, p1)
        out2.append(predict)
        count += 1
        if count == 50:
            break
    return out2

pred=predict(test_loader)
print(pred)
