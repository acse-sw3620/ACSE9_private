import torch,copy,argparse,csv
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# random seed
torch.manual_seed(15)
torch.cuda.manual_seed_all(15)
np.random.seed(15)
torch.backends.cudnn.deterministic = True





def main():
    ## ======================== load data ======================
    table = pd.read_csv("POD_coeffs_3900_new_grid_221_42.csv", header=None, dtype = float).values.astype(np.float16)
    data = copy.deepcopy(table[:,:-1].T)
    label = copy.deepcopy(table[:,1:].T)

    ## ======================== divide data ======================
    x_train, x_test, y_train, y_test = train_test_split(data,label, test_size=0.3 , random_state=2) # 70%用于训练，30%用于验证（测试）
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.FloatTensor)


    ## ======================== build MLP ======================
    # Generator，used to predict eigen-vector of next time level
    G = torch.nn.Sequential(
        torch.nn.Linear(10+10,30),   # eigen-vector + noise
        torch.nn.ReLU(),   
        torch.nn.Linear(30,15), 
        torch.nn.ReLU(),   
        torch.nn.Linear(15,10),   
    )
    # discriminator，used to tell whether the eigen-vector the output of generator or the real sample
    D = torch.nn.Sequential(
        torch.nn.Linear(10,6),   
        torch.nn.ReLU(),   
        torch.nn.Linear(6,3), 
        torch.nn.ReLU(),   
        torch.nn.Linear(3,1),
        nn.Sigmoid()   
    )


    loss_train = []
    loss_val = []

    ## ========================validate the model======================
    def valid(model,data,criterion):
        model.eval()
        z = torch.from_numpy(np.random.randn(data.shape[0], 10)).float() # random noise
        with torch.no_grad():   # do not record gradient information
            y_pred = model(torch.cat([z, data], dim=1))
            loss = criterion(y_pred, y_test)
            loss_val.append(loss.item())

    ## ======================== train model ======================
    loss_func_bce = nn.BCELoss()    
    loss_func_reg = nn.MSELoss()   
    opt_g = torch.optim.Adam(G.parameters(), lr=3e-4)   
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-4)
    epochs = 1000
    for epoch in range(epochs):
        # train discriminator
        for d in range(1):
            D.train()
            G.eval()
            # forward propagation
            z=torch.from_numpy(np.random.randn(x_train.shape[0], 10)).float()   # random noise
            d_real = D(y_train)     
            y_gen = G(torch.cat([z, x_train], dim=1))      
            d_gen = D(y_gen)        
            # calculate loss
            Dloss_real = loss_func_bce(d_real, torch.ones((x_train.shape[0],1))) 
            Dloss_gen = loss_func_bce(d_gen, torch.zeros((x_train.shape[0],1)))  
            Dloss = Dloss_real + Dloss_gen
            # backward propagation（only update parameters of discriminator）
            Dloss.backward()
            opt_d.step()
            opt_d.zero_grad()
            opt_g.zero_grad()
        # train generator
        for g in range(3):
            D.eval()
            G.train()
            # forward propagation
            z = torch.from_numpy(np.random.randn(x_train.shape[0], 10)).float() # random noise
            y_gen = G(torch.cat([z, x_train], dim=1))
            d_gen = D(y_gen)  
            # calculate loss
            Gloss_adventure = 0.3 * loss_func_bce(d_gen, torch.ones((x_train.shape[0],1))) 
            Gloss_regression = loss_func_reg(y_gen,y_train)
            Gloss = Gloss_regression + Gloss_adventure
            # backward propagation
            Gloss.backward()
            opt_g.step()
            opt_g.zero_grad()
            opt_d.zero_grad()
            loss_train.append(Gloss_regression.item())
            valid(G,x_test,loss_func_reg)
        D.eval()
        G.eval()
    


    ## ========================visualize the progress======================
    x = [i for i in range(epochs*3)]    
    plt.plot(x,loss_train, label='train')
    plt.plot(x,loss_val, label='val')
    plt.title('GAN_with_noise')
    plt.legend()
    plt.savefig(fname="result/GAN_with_noise.png")
    np.save('result/GAN_with_noise.npy',loss_train)   # save as .npy for further comparision
    print('GAN_with_noise——MSE of validation：',loss_val[-1])
    plt.show()
    
if __name__=='__main__':
    main()