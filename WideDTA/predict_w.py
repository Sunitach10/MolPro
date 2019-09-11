import data_w
from data_w import widedata
from data_w import*
# import train_w
# from train_w import*
trained_wmodel=torch.load('wide.pt')
dataset = widedata(ligand_path, protein_path,keys,motif_path,affinity_path)
train_loader, test_loader = load_splitset(dataset, .2)


def predict_w(model_,test):
    out21 = []
    count = 0
    for i,j in(test):
        m = i[0]
        m = m.reshape((1, 10, 10))
        p = i[1]
        p = p.reshape((1, 6729, 594))
        mt = i[2]
        mt = mt.reshape((1, 1076, 32))
        predict = model_(p, m, mt)
        out21.append(predict)
        count += 1
        if count == 50:
            break
    return out21

pred=predict_w(trained_wmodel,test_loader)
if __name__ == '__main__':
    print(pred)