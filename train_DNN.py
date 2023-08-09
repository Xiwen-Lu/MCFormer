import numpy as np
from tqdm import tqdm,trange
from torch import optim
from Model_DNN import *
from dataset import *
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torchvision import transforms

import yaml
import os

def test_currunt_model(settings):
    print("======Start Test DNN======")
    model = torch.load(os.path.join(settings['weights_dirpath']['dnn'], settings['weights_name']['dnn']))
    testdataset = ChannelDataset(settings['data_filepath']['test'],batch_length=100,is_slide=False,transforms=None)
    testLoader = DataLoader(testdataset, batch_size=1000, shuffle=False)

    score = 0
    total_num = 0
    model.eval()
    for batch_x, batch_y in testLoader:
        batch_x = normalize(batch_x.type(torch.FloatTensor), p=1, dim=1)
        X = batch_x.cuda().type(torch.cuda.FloatTensor)
        Y = np.array(batch_y).astype(int)
        Y_pre = model(X)
        Y_pre = Y_pre.cpu().detach().numpy()
        Y_pre = np.around(Y_pre)
        score += np.sum(Y == Y_pre)
        total_num += len(Y.reshape([-1]))

    score = score / total_num
    print("Test score: {:.6f}\nError rate: {:.6f}".format(score*100,1-score))
    return round(1-score,4)

def train(settings,nlayer=3):
    traindataset = ChannelDataset(settings['data_filepath']['train'],batch_length=100,is_slide=True,transforms=None)
    trainLoader = DataLoader(traindataset, batch_size=51200, shuffle=True)
    valdataset = ChannelDataset(settings['data_filepath']['val'],batch_length=100,is_slide=True,transforms=None)
    valLoader = DataLoader(valdataset,batch_size=51200,shuffle=False)

    in_dim=out_dim=100
    n_layers = nlayer
    hidden=128
    hidden_layers = [hidden for i in range(n_layers)]
    dropout=0.1
    model=DNN(in_dim=in_dim,hidden_layers=hidden_layers,out_dim=out_dim,dropout=dropout).cuda()
    # modelpath = os.path.join(settings['weights_dirpath'], settings['weights_name'])
    # model = torch.load(modelpath)

    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=20,verbose=True)
    criterion=nn.MSELoss()

    losses=[]
    val_accs=[]
    best_score = 0
    t = trange(500*n_layers)
    # without BatchNormalization and Dropout, then can drop .train() and .eval()
    # so that, it can avoid performance inconsistency caused by data normalization
    for epoch in t:
        model.train()
        for batch_x,batch_y in trainLoader:
            # batch_y=torch.from_numpy(batch_y).cuda()
            batch_y=batch_y.cuda()
            batch_y=batch_y.type(torch.cuda.FloatTensor)
            # batch_x=torch.from_numpy(batch_x).cuda()
            batch_x = normalize(batch_x.type(torch.FloatTensor),p=1,dim=1)
            batch_x=batch_x.cuda()
            batch_x=batch_x.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            pre_y=model(batch_x)
            loss=criterion(pre_y,batch_y)
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            model.eval()
            score = 0
            total_num = 0
            for batch_x,batch_y in valLoader:
                batch_x = normalize(batch_x.type(torch.FloatTensor), p=1, dim=1)
                X = batch_x.cuda().type(torch.cuda.FloatTensor)
                Y = np.array(batch_y).astype(int)
                Y_pre = model(X)
                Y_pre = Y_pre.cpu().detach().numpy()
                Y_pre = np.around(Y_pre)
                score += np.sum(Y == Y_pre)
                total_num += len(Y.reshape([-1]))

            score = score / total_num
            if score > best_score:
                best_score = score
                torch.save(model,os.path.join(settings['weights_dirpath']['dnn'],settings['weights_name']['dnn']))

        t.set_postfix({'loss':"{:.6f}".format(loss.item()),'val_acc':"{:.4f}".format(score*100)})
        # schedule.step(loss.item())
        losses.append(loss.item())
        val_accs.append(score*100)

    np.save(os.path.join(settings['log_dirpath']['train'],settings['weights_name']['dnn']),np.array(losses))
    np.save(os.path.join(settings['log_dirpath']['val'],settings['weights_name']['dnn']),np.array(val_accs))
    print("Best score: {:.6f}\nError rate: {:.6f}".format(best_score*100, 1 - best_score))

if __name__ == '__main__':
    nlayers = 1
    for v in [20,25,30,35,40,45,50]:
        settings_name = "./settings/settings_{}.yaml".format(v)
        settings = yaml.safe_load(open(settings_name))
        settings['weights_name']['dnn'] = "Model_DNN_{}layers_brownian_n_4000_r_1.5_v_{}.pth".format(nlayers, v)
        print("[v: {} \t nlayers: {}]--------".format(v, nlayers))
        train(settings=settings, nlayer=nlayers)
        _ = test_currunt_model(settings=settings)

    # test_currunt_model(yaml.safe_load(open("settings_45.yaml")))
