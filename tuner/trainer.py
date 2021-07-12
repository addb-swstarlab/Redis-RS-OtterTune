import os

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import utils

DEVICE = torch.device("cpu")

def mse_loss(targets,predict):
    return np.mean((predict.numpy()-targets.numpy())**2)

def mae_loss(targets,predict):
    return np.mean(np.absolute(predict[:,0]-targets[:,0]))

def train_single_epoch(model,trainDataloader,optimizer):
    train_loss = 0.0
    train_ACC = 0
    train_steps = 0
    model.train()
    for _ , batch in enumerate(tqdm(trainDataloader,desc="Iteration")):
        optimizer.zero_grad()
        knobs_with_info = batch[0].to(DEVICE)
        targets = batch[1].to(DEVICE)
        outputs = model(knobs_with_info)
        loss = F.mse_loss(outputs,targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_steps +=1

    return train_loss / len(trainDataloader), train_ACC

def train_twice_epoch(model,trainDataloader,optimizer):
    train_loss = 0.0
    train_ACC = 0
    train_steps = 0
    model.train()
    for _ , batch in enumerate(tqdm(trainDataloader,desc="Iteration")):
        optimizer.zero_grad()
        knobs_with_info = batch[0].to(DEVICE)
        targets = batch[1].to(DEVICE)

        outputs = model(knobs_with_info)

        loss = 0.
        for i, output in enumerate(outputs):
            loss += F.mse_loss(output.squeeze(1),targets[:,i])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_steps +=1

    return train_loss / len(trainDataloader), train_ACC

def eval_single_epoch(model, valDataloader):
    model.eval()
    val_loss = 0
    val_ACC = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(valDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            outputs = model(knobs_with_info)
            loss = F.mse_loss(outputs,targets)
            val_loss += loss.item()
    val_loss /= len(valDataloader)
    return val_loss, val_ACC

def eval_twice_epoch(model, valDataloader):
    model.eval()
    val_loss = 0
    val_ACC = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(valDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            outputs = model(knobs_with_info)
            loss = 0.
            for i, output in enumerate(outputs):
                loss += F.mse_loss(output.squeeze(1),targets[:,i])
            val_loss += loss.item()
    val_loss /= len(valDataloader)
    return val_loss, val_ACC

def test_single(model, testDataloader, scaler_y):
    test_loss, test_mae = 0, 0
    model.eval()
    test_losses, mae_losses = [], []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = scaler_y.inverse_transform(batch[1].to(DEVICE))
            outputs = scaler_y.inverse_transform(model(knobs_with_info))
            for i in range(2):
                mse = mse_loss(targets[:,i],outputs[:,i])
                mae = mae_loss(targets[:,i],outputs[:,i])
                test_losses.append(mse.item())
                mae_losses.append(mae.item())
                test_loss+=mse.item()
                test_mae+=mae.item()
    test_losses = np.array(test_losses)/len(testDataloader)
    test_mae = np.array(mae_losses)/len(testDataloader)
    return test_losses, test_mae

# def test_twice(model, testDataloader, scaler_y):
#     test_loss, test_mae = 0, 0
#     model.eval()
#     with torch.no_grad():
#         for _, batch in enumerate(tqdm(testDataloader,desc="Iteration")):
#             knobs_with_info = batch[0].to(DEVICE)
#             targets = scaler_y.inverse_transform(batch[1].to(DEVICE))
#             outputs = scaler_y.inverse_transform(model(knobs_with_info))
#             loss = 0.
#             mae = 0.
#             for i, output in enumerate(outputs):
#                 loss += F.mse_loss(output.squeeze(1),targets[:,i])
#                 mae += np.mean(np.absolute(output.squeeze(1).numpy()-targets[:,i].numpy()))
#             test_loss += loss.item()
#             test_mae += mae
#     test_loss /=len(testDataloader)
#     test_mae /=len(testDataloader)
#     return test_loss, test_mae

def test_twice(model, testDataloader, scaler_y):
    test_loss, test_mae = 0, 0
    model.eval()
    test_losses, mae_losses = [], []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = scaler_y.inverse_transform(batch[1].to(DEVICE))
            throughput, latency = model(knobs_with_info)
            outputs = torch.cat([throughput,latency],axis=1)
            outputs = scaler_y.inverse_transform(outputs)
            for i in range(2):
                mse = mse_loss(targets[:,i],outputs[:,i])
                mae = mae_loss(targets[:,i],outputs[:,i])
                test_losses.append(mse.item())
                mae_losses.append(mae.item())
                test_loss+=mse.item()
                test_mae+=mae.item()
    test_losses = np.array(test_losses)/len(testDataloader)
    test_mae = np.array(mae_losses)/len(testDataloader)
    return test_losses, test_mae

def train(model, trainDataloader, valDataloader, testDataloader, optimizer, scaler_y, opt, logger):
    val_losses = []
    test_losses = []
    model_save_path = utils.make_date_dir("./model_save")
    logger.info("Model save path : {}".format(model_save_path))
    logger.info("Learning Rate : {}".format(opt.lr))

    if opt.model_mode == 'single':
        train_epoch = train_single_epoch
        eval_epoch = eval_single_epoch
        test = test_single
    elif opt.model_mode == 'twice':
        train_epoch = train_twice_epoch
        eval_epoch = eval_twice_epoch
        test = test_twice

    best_loss = float('inf')
    patience = 0

    #scheduler = LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch:0.95**epoch,last_epoch=-1,verbose=False)
    for epoch in range(int(opt.n_epochs)):
        patience +=1
        logger.info("====================================Train====================================")
        train_loss, _ = train_epoch(model,trainDataloader,optimizer)
        logger.info("[Train Epoch {}] train Loss : {}".format(epoch+1,train_loss))

        logger.info("====================================Val====================================")
        val_loss, _ = eval_epoch(model,valDataloader)
        logger.info("[Eval Epoch {}] val Loss : {}".format(epoch+1,val_loss))

        logger.info("====================================Test====================================")
        test_loss, test_mae = test(model, testDataloader, scaler_y)
        th_loss, la_loss = test_loss
        th_mae_loss, la_mae_loss = test_mae
        logger.info("[Epoch {}] Test_throughput_Loss: {}, Test_latency_Loss: {} , Test_throughput_MAE_Loss: {}, Test_latency_MAE_Loss: {}".format(epoch+1, test_loss[0], test_loss[1], test_mae[0], test_mae[1]))
        test_loss = sum(test_loss)
        test_mae = sum(test_mae)
        if test_loss < best_loss:
            torch.save(model.state_dict(),os.path.join(model_save_path,"model_"+str(epoch+1)+".pt"))
            best_th_loss, best_la_loss, best_th_mae_loss, best_la_mae_loss  = th_loss, la_loss, th_mae_loss, la_mae_loss
            best_loss = test_loss
            patience = 0
            best_epoch = epoch+1
        if patience == 5:
            break
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        
    return best_epoch, best_th_loss, best_la_loss, best_th_mae_loss, best_la_mae_loss, model_save_path