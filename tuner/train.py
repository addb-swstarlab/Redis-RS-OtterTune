# -*- coding: utf-8 -*-
"""
Train the model
"""

import os
import sys
import utils
import knobs
import argparse
from glob import glob
sys.path.append('../')
##TODO: we can use environment after...
import copy

from models.steps import (metricSimplification, knobsRanking, configuration_recommendation)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tencent', action='store_true', help='Use Tencent Server')
    parser.add_argument('--params', type=str, default='', help='Load existing parameters')
    parser.add_argument('--target', type=int, default= 20, help='Target Workload')
    parser.add_argument('--persistence', type=str, choices=["RDB","AOF"], default='RDB', help='Choose Persistant Methods')
    parser.add_argument("--db",type=str, choices=["redis","rocksdb"], default='redis', help="DB type")
    parser.add_argument("--rki",type=str, default='lasso', help = "knob_identification mode")
    parser.add_argument("--gp", type=str, default="numpy")
    
    opt = parser.parse_args()
    DATA_PATH = "../data/redis_data"
    PATH=None

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

    #Target workload loading
    logger.info("Target workload name is {}".format(opt.target))
    target_DATA_PATH = "../data/redis_data/workload{}".format(opt.target)
    
    knobs_path = os.path.join(DATA_PATH, "configs")
    if opt.persistence == "RDB":
        knob_data, _ = knobs.load_knobs(knobs_path)
    elif opt.persistence == "AOF":
        _, knob_data = knobs.load_knobs(knobs_path)

    logger.info("Finish Load Knob Data")

    internal_metric_datas = {}
    external_metric_datas = {}

    # len()-1 because of configs dir
    for i in range(1,len(os.listdir(DATA_PATH))):
    #for i in range(1,10):
        if opt.target == i:
            target_external_data = knobs.load_metrics(metric_path = os.path.join(target_DATA_PATH ,f"result_{opt.persistence.lower()}_external_{i}.csv"),
                                                labels = knob_data['rowlabels'],
                                                metrics = ['Totals_Ops/sec', 'Totals_p99_Latency'])
        else:
            internal_metric_data, dict_le_in = knobs.load_metrics(metric_path = os.path.join(DATA_PATH,f'workload{i}',f'result_{opt.persistence.lower()}_internal_{i}.csv'),
                                                            labels = knob_data['rowlabels'])
            
            external_metric_data, _ = knobs.load_metrics(metric_path = os.path.join(DATA_PATH,f'workload{i}',f'result_{opt.persistence.lower()}_external_{i}.csv'),
                                                labels = knob_data['rowlabels'],
                                                metrics = ['Totals_Ops/sec', 'Totals_p99_Latency'])
            internal_metric_datas[f'workload{i}'] = internal_metric_data['data']
            external_metric_datas[f'workload{i}'] = external_metric_data['data']

    internal_metric_datas['columnlabels'] = internal_metric_data['columnlabels']
    internal_metric_datas['rowlabels'] = internal_metric_data['rowlabels']
    logger.info("Finish Load Internal and External Metrics Data")

    """
    workload{2~18} = workload datas composed of different key(workload2, workload3, ...) [N of configs, N of columnlabels]
    columnlabels  = Internal Metric names
    rowlabels = Index for Workload data

    internal_metric_datas = {
        'workload{2~18} except target(1)'=array([[1,2,3,...], [2,3,4,...], ...[]])
        'columnlabels'=array(['IM_1', 'IM_2', ...]),
        'rowlabels'=array([1, 2, ..., 10000])}
    """

    aggregated_IM_data = knobs.aggregateInternalMetrics(internal_metric_datas)

    """
    data = concat((workload2,...,workload18)) length = 10000 * N of workload
    columnlabels  = same as internal_metric_datas's columnlabels
    rowlabels = same as internal_metric_datas's rowlabels

    aggregated_IM_data = {
        'data'=array([[1,2,3,...], [2,3,4,...], ...[]])
        'columnlabels'=array(['IM_1', 'IM_2', ...]),
        'rowlabels'=array([1, 2, ..., 10000])}
    
    """

    logger.info("====================== metricSimplification ====================")
    pruned_metrics = metricSimplification(aggregated_IM_data, logger)
    logger.info("Done pruning metrics for workload {} (# of pruned metrics: {}).\n\n""Pruned metrics: {}\n".format(opt.persistence, len(pruned_metrics),pruned_metrics))
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
    logger.info("\n\n====================== run_knob_identification ====================")
    logger.info("use mode = {}".format(opt.rki))
    ranked_knobs = knobsRanking(knob_data = rank_knob_data,
                                metric_data = ranked_metric_data,
                                mode = opt.rki,
                                logger = logger)
    logger.info("Done ranking knobs for workload {} (# ranked knobs: {}).\n\n"
                 "Ranked knobs: {}\n".format(opt.persistence, len(ranked_knobs), ranked_knobs))

    assert False
    ### WORKLOAD MAPPING ###
    ## TODO: ...                 



    ### RECOMMENDATION STAGE ###
    ##TODO: choose k like incremental 4, 8, 16, ...
    top_ks = range(4,13)
    best_recommend = -float('inf')
    best_topk = None
    best_conf_map = None
    for top_k in top_ks:        
        logger.info("\n\n================ The number of TOP knobs ===============")
        logger.info(top_k)

        ranked_test_knob_data = utils.get_ranked_knob_data(ranked_knobs, test_knob_data, top_k)
        
        ## TODO: params(GP option) and will offer opt all
        FIN,recommend,conf_map = configuration_recommendation(ranked_test_knob_data,test_external_data, logger, opt.gp, opt.db, opt.persistence)

        if recommend > best_recommend and FIN:
            best_recommend = recommend
            best_topk = top_k
            best_conf_map = conf_map
    logger.info("Best top_k")
    logger.info(best_topk)
    print(best_topk)
    utils.convert_dict_to_conf(best_conf_map, opt.persistence)

    print("END TRAIN")