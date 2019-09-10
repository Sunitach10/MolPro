import data_w
from data_w import*
import model_w
from model_w import*
import train_w
from train_w import*
def predict_w(test):
    out21 = []
    count = 0
    for i,j in(test):
        m = i[0]
        m = m.reshape((1, 10, 10))
        p = i[1]
        p = p.reshape((1, 6729, 594))
        mt = i[2]
        mt = mt.reshape((1, 1076, 32))
        predict = modelw(p, m, mt)
        out21.append(predict)
        count += 1
        if count == 50:
            break
    return out21

pred=predict_w(test_loader)
if __name__ == '__main__':
    print(pred)