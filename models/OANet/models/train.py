# train_train_test_split

import torch
import pandas as pd
import numpy as np
import os
# from sklearn.model_selection import StratifiedKFold
from models.OANet.models.neural_network_train import NeuralModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error 

def train_Net(logger, data, METRIC, MODE, batch_size, lr, epochs, hidden_dim, group_dim, WK_NUM, dot, EX_NUM=4, lamb=0.1):
    df_pred = pd.DataFrame(columns=("METRIC", "r2", 'pcc', "ci", "MSE"))

    k_r2 = 0
    k_pcc = 0
    k_ci = 0
    k_MSE = 0
    cnt = 0

    X = data.iloc[:,EX_NUM:]
    Y = data[[METRIC]]

    X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1004)
      

    # TODO: scale
    scaler_X = MinMaxScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(y_tr)
        
    norm_X_tr = torch.Tensor(scaler_X.transform(X_tr)).cuda()
    norm_X_te = torch.Tensor(scaler_X.transform(X_te)).cuda()
    norm_y_tr = torch.Tensor(scaler_y.transform(y_tr)).cuda()
    norm_y_te = torch.Tensor(scaler_y.transform(y_te)).cuda()

    model = NeuralModel(logger, mode=MODE, batch_size=batch_size, lr=lr, epochs=epochs, 
                            input_dim=norm_X_tr.shape[-1], hidden_dim=hidden_dim, output_dim=norm_y_tr.shape[-1],
                            group_dim=group_dim, wk_num=WK_NUM, dot=dot, lamb=lamb)
    X = (norm_X_tr, norm_X_te)
    y = (norm_y_tr, norm_y_te)
    model.fit(X, y)
    outputs = model.predict(norm_X_te)

    true = norm_y_te.cpu().detach().numpy().squeeze()
    pred = outputs.cpu().detach().numpy().squeeze()

    r2_res = r2_score(true, pred)

    pcc_res, _ = pearsonr(true, pred)
    ci_res = concordance_index(true, pred)

    MSE_res = mean_squared_error(true, pred)

    cnt += 1
    # print(f"-------{cnt}-------")
    print(f"--------results-------")
    print(f"r2  score = {r2_res:.4f}")
    print(f"pcc score = {pcc_res:.4f}")
    print(f"ci  score = {ci_res:.4f}")
    print(f"MSE score = {MSE_res:.4f}")


    k_r2 += r2_res
    k_pcc += pcc_res
    k_ci += ci_res
    k_MSE += MSE_res

    del norm_X_tr, norm_X_te, norm_y_tr, norm_y_te
    torch.cuda.empty_cache()
    
    # print(f"-------Mean of results-------")
    # print(f"r2  is {k_r2/cnt}")
    # print(f"pcc is {k_pcc/cnt}")
    # print(f"ci  is {k_ci/cnt}")
    # print(f"MSE is {MSE_res/cnt}")
    
    score = [ (METRIC, k_r2/cnt, k_pcc/cnt, k_ci/cnt, MSE_res/cnt) ]
    ex = pd.DataFrame(score, columns=["METRIC", "r2", 'pcc', "ci", "MSE"])
    df_pred = pd.concat(([df_pred, ex]), ignore_index=True )
    return k_r2/cnt, k_pcc/cnt, k_ci/cnt, MSE_res/cnt, true, pred, df_pred
