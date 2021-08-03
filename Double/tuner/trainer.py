import os

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import utils

DEVICE = torch.device("cpu")


def train_double_epoch(model,trainDataloader,optimizer):
    train_loss = 0.0
    train_ACC = 0
    train_steps = 0
    model.train()
    for _ , batch in enumerate(tqdm(trainDataloader,desc="Iteration")):
        optimizer.zero_grad()
        knobs_with_info = batch[0].to(DEVICE)
        targets = batch[1].to(DEVICE)

        outputs = model(knobs_with_info)
        loss = F.mse_loss(outputs,targets.unsqueeze(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_steps +=1

    return train_loss / len(trainDataloader), train_ACC


def eval_double_epoch(model, valDataloader):
    model.eval()
    val_loss = 0
    val_ACC = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(valDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            outputs = model(knobs_with_info)
            loss = F.mse_loss(outputs,targets.unsqueeze(-1))
            val_loss += loss.item()
    val_loss /= len(valDataloader)
    return val_loss, val_ACC


def test_double(model, testDataloader, scaler_y):
    test_loss, test_mae = 0, 0
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            target = batch[1].to(DEVICE)
            output = model(knobs_with_info)
            target = torch.tensor(scaler_y.inverse_transform(target))
            output = torch.tensor(scaler_y.inverse_transform(output))
            loss = F.mse_loss(output,target.unsqueeze(-1))
            mae = np.mean(np.absolute(output.numpy()-target.numpy()))

            test_loss += loss.item()
            test_mae += mae

    test_loss /= len(testDataloader)
    test_mae /= len(testDataloader)
    return test_loss, test_mae


def train(model, trainDataloader, valDataloader, testDataloader, optimizer, scaler_y, opt, logger,model_save_path, index):
    val_losses = []
    test_losses = []

    train_epoch = train_double_epoch
    eval_epoch = eval_double_epoch
    test = test_double

    best_loss = float('inf')
    best_epoch = 0
    patience = 0

    #scheduler = LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch:0.95**epoch,last_epoch=-1,verbose=False)
    best_epoch = 0.
    best_loss= float('inf')
    best_mae = 0.
    for i, model in enumerate(model.items()):
        if i!=index:
            continue
        model_name, model = model
        logger.info("\n\n===================================={}====================================".format(model_name))
        tmp_val_losses = []
        tmp_test_losses = []
        for epoch in range(int(opt.n_epochs)):
            patience +=1
            logger.info("====================================Train====================================")
            train_loss, _ = train_epoch(model,trainDataloader,optimizer[model_name])
            logger.info("[Train Epoch {}] train Loss : {}".format(epoch+1,train_loss))

            logger.info("====================================Val====================================")
            val_loss, _ = eval_epoch(model,valDataloader)
            logger.info("[Eval Epoch {}] val Loss : {}".format(epoch+1,val_loss))

            logger.info("====================================Test====================================")
            test_loss, test_mae = test(model, testDataloader, scaler_y)
            
            logger.info("[Epoch {}] Test_Loss: {}, Test_MAE : {}".format(epoch+1, test_loss, test_mae))

            if test_loss < best_loss:
                #double needs save naming
                torch.save(model.state_dict(),os.path.join(model_save_path,"{}_{}.pt".format(model_name,str(epoch+1))))
                best_loss = test_loss
                best_mae = test_mae
                best_epoch = epoch+1
                patience = 0
            if patience == 5:
                break
            
            tmp_val_losses.append(val_loss)
            tmp_test_losses.append(test_loss)

        val_losses.extend(tmp_val_losses)
        test_losses.extend(tmp_test_losses)

    return best_epoch, best_loss, best_mae