# -*- coding: utf-8 -*-
"""
Train the model
"""
import os
import sys
import copy
import logging
import argparse
from random import randint

import numpy as np
import torch

import utils

from trainer import train
from config import Config

sys.path.append('../')
from models.steps import (dataPreprocessing, metricSimplification, knobsRanking, prepareForTraining, )


parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, default=1)
parser.add_argument('--persistence', type = str, choices = ["RDB","AOF"], default = 'RDB', help='Choose Persistant Methods')
parser.add_argument("--db",type = str, choices = ["redis","rocksdb"], default = 'redis', help="DB type")
parser.add_argument("--cluster",type=str, choices = ['k-means','ms','gmm'],default = 'ms')
parser.add_argument("--rki",type = str, default = 'lasso', help = "knob_identification mode")
parser.add_argument("--topk",type = int, default = 4, help = "Top # knobs")
parser.add_argument("--n_epochs",type = int, default = 100, help = "Train # epochs with model")
parser.add_argument("--lr", type = float, default = 1e-5, help = "Learning Rate")
parser.add_argument("--model_mode", type = str, default = 'single', help = "model mode")

opt = parser.parse_args()
DATA_PATH = "../data/redis_data"
DEVICE = torch.device("cpu")

if not os.path.exists('save_knobs'):
    os.mkdir('save_knobs')

#expr_name = 'train_{}'.format(utils.config_exist(opt.persistence))

def main(opt: argparse , logger: logging, log_dir: str) -> Config:
    #Target workload loading
    logger.info("====================== {} mode ====================\n".format(opt.persistence))

    logger.info("Target workload name is {}".format(opt.target))

    knob_data, aggregated_IM_data, aggregated_EM_data, target_external_data = dataPreprocessing(opt.target, opt.persistence,logger)

    logger.info("====================== Metrics_Simplification ====================\n")
    pruned_metrics = metricSimplification(aggregated_IM_data, logger, opt)
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
    logger.info("====================== Run_Knobs_Ranking ====================\n")
    logger.info("use mode = {}".format(opt.rki))
    ranked_knobs = knobsRanking(knob_data = rank_knob_data,
                                metric_data = ranked_metric_data,
                                mode = opt.rki,
                                logger = logger)
    logger.info("Done ranking knobs for workload {} (# ranked knobs: {}).\n\n"
                 "Ranked knobs: {}\n".format(opt.persistence, len(ranked_knobs), ranked_knobs))

    top_k = opt.topk
    """
    top_k_knobs : dict()
    """
    top_k_knobs = utils.get_ranked_knob_data(ranked_knobs, knob_data, top_k)
    knob_save_path = utils.make_date_dir('./save_knobs')
    logger.info("Knob save path : {}".format(knob_save_path))
    logger.info("Choose Top {} knobs : {}".format(top_k,top_k_knobs['columnlabels']))
    np.save(os.path.join(knob_save_path,'knobs_{}.npy'.format(top_k)),np.array(top_k_knobs['columnlabels']))

    model, optimizer, trainDataloader, valDataloader, testDataloader, scaler_y = prepareForTraining(opt, top_k_knobs, aggregated_EM_data, target_external_data)
    
    logger.info("====================== {} Pre-training Stage ====================\n".format(opt.model_mode))

    best_epoch, best_th_loss, best_la_loss, best_th_mae_loss, best_la_mae_loss, model_path = train(model, trainDataloader, valDataloader, testDataloader, optimizer, scaler_y, opt, logger)
    if opt.model_mode in ['single', 'twice']:
        logger.info("\n\n[Best Epoch {}] Best_th_Loss : {} Best_la_Loss : {} Best_th_MAE : {} Best_la_MAE : {}".format(best_epoch, best_th_loss, best_la_loss, best_th_mae_loss, best_la_mae_loss))
    elif opt.model_mode == 'double':
        for name in best_epoch.keys():
            logger.info("\n\n[{} Best Epoch {}] Best_Loss : {} Best_MAE : {}".format(name, best_epoch[name], best_loss[name], best_mae[name]))
    
    config = Config(opt.persistence,opt.db,opt.cluster,opt.rki,opt.topk,opt.model_mode,opt.n_epochs,opt.lr)
    config.save_results(opt.target, best_epoch, best_th_loss, best_la_loss, best_th_mae_loss, best_la_mae_loss, model_path, log_dir)

    return config

if __name__ == '__main__':
    print("======================MAKE LOGGER====================")
    logger, log_dir = utils.get_logger(os.path.join('./logs'))
    '''
        internal_metrics, external_metrics, knobs
        metric_data : internal metrics
        knobs_data : configuration knobs
    '''
    try:
        main(opt, logger, log_dir)
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()

    """
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