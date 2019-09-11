import data
from data import*
# import model
# from model import*
# import train
# from train import*
trained_model = torch.load('deep.pt')
dataset = NumbersDataset(ligand_path, protein_path, affinity_path)
train_loader, test_loader = load_splitset(dataset, .1)
def predict(model3,test):
    out2 = []
    count = 0
    for i,j in(test):
        m1 = i[0]
        m1 = m1.reshape((1, 62, 50))
        p1 = i[1]
        p1 = p1.reshape((1, 25, 600))
        predict = model3(m1, p1)
        # _, preds_tensor = torch.max(predict, 1)
        out2.append(predict)
        # out2.append(preds_tensor)
        count += 1
        if count == 5:
            break
    return out2

pred=predict(trained_model,test_loader)
print(pred)
