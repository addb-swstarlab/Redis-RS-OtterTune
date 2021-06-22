# -*- coding: utf-8 -*-
"""
Train the model
"""

import os
import sys
import argparse
import copy

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import utils

sys.path.append('../')
##TODO: we can use environment after...

from models.steps import (dataPreprocessing, metricSimplification, knobsRanking, prepareForTraining, )


parser = argparse.ArgumentParser()
parser.add_argument('--params', type = str, default = '', help='Load existing parameters')
parser.add_argument('--target', type = int, default =  1, help='Target Workload')
parser.add_argument('--persistence', type = str, choices = ["RDB","AOF"], default = 'RDB', help='Choose Persistant Methods')
parser.add_argument("--db",type = str, choices = ["redis","rocksdb"], default = 'redis', help="DB type")
parser.add_argument("--rki",type = str, default = 'lasso', help = "knob_identification mode")
parser.add_argument("--gp", type = str, default = "numpy")
parser.add_argument("--topk",type = int, default = 4, help = "Top # knobs")
parser.add_argument("--n_epochs",type = int, default = 50, help = "Train # epochs with model")
parser.add_argument("--lr", type = float, default = 1e-6, help = "Learning Rate")

opt = parser.parse_args()
DATA_PATH = "../data/redis_data"
DEVICE = torch.device("cpu")


if not os.path.exists('save_knobs'):
    os.mkdir('save_knobs')

#expr_name = 'train_{}'.format(utils.config_exist(opt.persistence))

print("======================MAKE LOGGER====================")
logger, log_dir = utils.get_logger(os.path.join('./logs'))

logger.info("#==============================Data PreProcessing Stage=================================")
'''
    internal_metrics, external_metrics, knobs
    metric_data : internal metrics
    knobs_data : configuration knobs
'''

def train_epoch(model,trainDataloader,optimizer):
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

def eval_epoch(model, valDataloader):
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

def test(model, testDataloader):
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


def train(model, trainDataloader, valDataloader, testDataloader, optimizer):
    val_losses = []
    test_losses = []
    model_save_path = utils.make_date_dir("./model_save")
    logger.info("Model save path : {}".format(model_save_path))

    best_loss = float('inf')
    best_acc = 0
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
        test_loss, test_mae = test(model,testDataloader)
        
        logger.info("[Epoch {}] Test_Loss: {}, Test_MAE : {}".format(epoch+1, test_loss, test_mae))

        if test_loss < best_loss:
            torch.save(model.state_dict(),os.path.join(model_save_path,"model_"+str(epoch+1)+".pt"))
            best_loss = test_loss
            patience = 0
        if patience == 5:
            break

        val_losses.append(val_loss)
        test_losses.append(test_loss)

    return val_losses, test_losses


if __name__ == '__main__':

    #Target workload loading
    logger.info("Target workload name is {}".format(opt.target))

    knob_data, aggregated_IM_data, aggregated_EM_data, target_external_data = dataPreprocessing(opt.target, opt.persistence,logger)

    logger.info("====================== Metrics_Simplification ====================")
    pruned_metrics = metricSimplification(aggregated_IM_data, logger)
    logger.info("Done pruning metrics for workload {} (# of pruned metrics: {}).\n\n""Pruned metrics: {}\n".format(opt.persistence, len(pruned_metrics), pruned_metrics))
    metric_idxs = [i for i, metric_name in enumerate(aggregated_IM_data['columnlabels']) if metric_name in pruned_metrics]
    ranked_metric_data = {
        'data' : aggregated_IM_data['data'][:,metric_idxs],
        'rowlabels' : copy.deepcopy(aggregated_IM_data['rowlabels']),
        'columnlabels' : [aggregated_IM_data['columnlabels'][i] for i in metric_idxs]
    }
    """
        For example,
            pruned_metrics : ['allocator_rss_bytes', 'rss_overhead_bytes', 'used_memory_dataset', 'rdb_last_cow_size']
    """
    ### KNOBS RANKING STAGE ###
    rank_knob_data = copy.deepcopy(knob_data)
    logger.info("\n\n====================== Run_Knobs_Ranking ====================")
    logger.info("use mode = {}".format(opt.rki))
    ranked_knobs = knobsRanking(knob_data = rank_knob_data,
                                metric_data = ranked_metric_data,
                                mode = opt.rki,
                                logger = logger)
    logger.info("Done ranking knobs for workload {} (# ranked knobs: {}).\n\n"
                 "Ranked knobs: {}\n".format(opt.persistence, len(ranked_knobs), ranked_knobs))

    top_k = opt.topk
    top_k_knobs = utils.get_ranked_knob_data(ranked_knobs, knob_data, top_k)

    model, optimizer, trainDataloader, valDataloader, testDataloader = prepareForTraining(opt.target, opt.lr, top_k_knobs, aggregated_EM_data, target_external_data)

    val_losses, test_losses = train(model, trainDataloader, valDataloader, testDataloader, optimizer)

    """
    #### Pre-train Stage(Rename)
    1. Dataset 내에서 scaler가 잘 작동하는지 print
    2. Dataloader 다시 확인하기
    3. Dense
    4. 쥬피터에서 train으로 코드 이동
    ~제주도
    #### Recommendation
    5. GA
    6. 결과를 config로 만드는 방법
    #### Clustering 기법들 비교
    1. 중심값
    2. 파라미터
    #### Problem
    현재는 한번에 두 개를 동시에 맞추는 모델임.
    1. 각각의 Dense가 예측을 하게 함
        -> 편차가 생김(계수를 부여)
    2. 아예 모델을 2개 만드는 거임
        -> EM간의 trade-off를 못해
        -> 두 개의 config 추천이 나옴
    """
    
    # ### RECOMMENDATION STAGE ###
    # ##TODO: choose k like incremental 4, 8, 16, ...
    # top_ks = range(4,13)
    # best_recommend = -float('inf')
    # best_topk = None
    # best_conf_map = None
    # for top_k in top_ks:        
    #     logger.info("\n\n================ The number of TOP knobs ===============")
    #     logger.info(top_k)

    #     ranked_test_knob_data = utils.get_ranked_knob_data(ranked_knobs, test_knob_data, top_k)
        
    #     ## TODO: params(GP option) and will offer opt all
    #     FIN,recommend,conf_map = configuration_recommendation(ranked_test_knob_data,test_external_data, logger, opt.gp, opt.db, opt.persistence)

    #     if recommend > best_recommend and FIN:
    #         best_recommend = recommend
    #         best_topk = top_k
    #         best_conf_map = conf_map
    # logger.info("Best top_k")
    # logger.info(best_topk)
    # print(best_topk)
    # utils.convert_dict_to_conf(best_conf_map, opt.persistence)

    # print("END TRAIN")