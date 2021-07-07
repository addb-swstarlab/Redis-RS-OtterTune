import os

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import utils

DEVICE = torch.device("cpu")

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

def train_double_epoch(model,trainDataloader,optimizer,index=0):
    train_loss = 0.0
    train_ACC = 0
    train_steps = 0
    model.train()
    for _ , batch in enumerate(tqdm(trainDataloader,desc="Iteration")):
        optimizer.zero_grad()
        knobs_with_info = batch[0].to(DEVICE)
        targets = batch[1].to(DEVICE)

        outputs = model(knobs_with_info)
        loss = F.mse_loss(outputs,targets[:,index].unsqueeze(-1))
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

def eval_double_epoch(model, valDataloader, index=0):
    model.eval()
    val_loss = 0
    val_ACC = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(valDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            outputs = model(knobs_with_info)
            loss = F.mse_loss(outputs,targets[:,index].unsqueeze(-1))
            val_loss += loss.item()
    val_loss /= len(valDataloader)
    return val_loss, val_ACC

def test_single(model, testDataloader):
    test_loss, test_mae = 0, 0
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            outputs = model(knobs_with_info)
            loss = F.mse_loss(outputs,targets)
            mae = np.mean(np.absolute(outputs.numpy()-targets.numpy()))
            test_loss += loss.item()
            test_mae += mae
    return test_loss, test_mae

def test_twice(model, testDataloader):
    test_loss, test_mae = 0, 0
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            outputs = model(knobs_with_info)
            loss = 0.
            mae = 0.
            for i, output in enumerate(outputs):
                loss += F.mse_loss(output.squeeze(1),targets[:,i])
                mae += np.mean(np.absolute(output.squeeze(1).numpy()-targets[:,i].numpy()))
            test_loss += loss.item()
            test_mae += mae
    return test_loss, test_mae

def test_double(model, testDataloader, index=0):
    test_loss, test_mae = 0, 0
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testDataloader,desc="Iteration")):
            knobs_with_info = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            outputs = model(knobs_with_info)
            loss = F.mse_loss(outputs,targets[:,index].unsqueeze(-1))
            mae = np.mean(np.absolute(outputs.numpy()-targets[:,index].unsqueeze(-1).numpy()))
            test_loss += loss.item()
            test_mae += mae
    return test_loss, test_mae


def train(model, trainDataloader, valDataloader, testDataloader, optimizer, opt, logger):
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
    elif opt.model_mode == 'double':
        train_epoch = train_double_epoch
        eval_epoch = eval_double_epoch
        test = test_double

    best_loss = float('inf')
    best_epoch = 0
    patience = 0

    #scheduler = LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch:0.95**epoch,last_epoch=-1,verbose=False)
    if opt.model_mode in ['single', 'twice']:
        for epoch in range(int(opt.n_epochs)):
            patience +=1
            logger.info("====================================Train====================================")
            train_loss, _ = train_epoch(model,trainDataloader,optimizer)
            logger.info("[Train Epoch {}] train Loss : {}".format(epoch+1,train_loss))

            logger.info("====================================Val====================================")
            val_loss, _ = eval_epoch(model,valDataloader)
            logger.info("[Eval Epoch {}] val Loss : {}".format(epoch+1,val_loss))

            logger.info("====================================Test====================================")
            test_loss, test_mae = test(model,testDataloader)
            
            logger.info("[Epoch {}] Test_Loss: {}, Test_MAE : {}".format(epoch+1, test_loss, test_mae))

            if test_loss < best_loss:
                torch.save(model.state_dict(),os.path.join(model_save_path,"model_"+str(epoch+1)+".pt"))
                best_loss = test_loss
                best_mae = test_mae
                patience = 0
                best_epoch = epoch+1
            if patience == 5:
                break

            val_losses.append(val_loss)
            test_losses.append(test_loss)
    elif opt.model_mode == 'double':
        best_epoch = dict(zip(list(model.keys()),[0.,0.]))
        best_loss= dict(zip(list(model.keys()),[float('inf'),float('inf')]))
        best_mae = dict(zip(list(model.keys()),[0.,0.]))
        for i, model in enumerate(model.items()):
            model_name, model = model
            logger.info("\n\n===================================={}====================================".format(model_name))
            tmp_val_losses = []
            tmp_test_losses = []
            for epoch in range(int(opt.n_epochs)):
                patience +=1
                logger.info("====================================Train====================================")
                train_loss, _ = train_epoch(model,trainDataloader,optimizer[model_name],index=i)
                logger.info("[Train Epoch {}] train Loss : {}".format(epoch+1,train_loss))

                logger.info("====================================Val====================================")
                val_loss, _ = eval_epoch(model,valDataloader,index=i)
                logger.info("[Eval Epoch {}] val Loss : {}".format(epoch+1,val_loss))

                logger.info("====================================Test====================================")
                test_loss, test_mae = test(model,testDataloader,index=i)
                
                logger.info("[Epoch {}] Test_Loss: {}, Test_MAE : {}".format(epoch+1, test_loss, test_mae))

                if test_loss < best_loss[model_name]:
                    #double needs save naming
                    torch.save(model.state_dict(),os.path.join(model_save_path,"{}_{}.pt".format(model_name,str(epoch+1))))
                    best_loss[model_name] = test_loss
                    best_mae[model_name] = test_mae
                    best_epoch[model_name] = epoch+1
                    patience = 0
                if patience == 5:
                    break
                
                tmp_val_losses.append(val_loss)
                tmp_test_losses.append(test_loss)

            val_losses.extend(tmp_val_losses)
            test_losses.extend(tmp_test_losses)

    return best_epoch, best_loss, best_mae, model_save_path