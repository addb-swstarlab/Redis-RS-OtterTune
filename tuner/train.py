# -*- coding: utf-8 -*-
"""
Train the model
"""

import os
import sys
import utils
import argparse
sys.path.append('../')
import copy
from sklearn.model_selection import train_test_split

from models.steps import (run_workload_characterization, run_knob_identification, configuration_recommendation)

DATA_PATH = "../data"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tencent', action='store_true', help='Use Tencent Server')
    parser.add_argument('--params', type=str, default='', help='Load existing parameters')
    parser.add_argument('--target', type=int, default=1, help='Workload type')
    parser.add_argument('--persistence', type=str, choices=["RDB","AOF"], default='RDB', help='Choose Persistant Methods')
    parser.add_argument("--db",type=str, choices=["redis","rocksdb"], default='redis', help="DB type")
    parser.add_argument("--rki",type=str, default='XGB', help = "knob_identification mode")
    parser.add_argument("--gp", type=str, default="numpy")

    opt = parser.parse_args()

    PATH=None

    if not os.path.exists('logs'):
        os.mkdir('logs')

    if not os.path.exists('save_knobs'):
        os.mkdir('save_knobs')

    expr_name = 'train_{}'.format(utils.config_exist(opt.persistence))


    print("======================MAKE LOGGER at %s====================" % expr_name)
    ##TODO: save log
    logger = utils.Logger(
        name=opt.persistence,
        log_file='logs/{}/{}.log'.format(opt.persistence, expr_name)
    )

    #==============================Data PreProcessing Stage=================================
    # Read sample-metric matrix, need knob name(label)
    '''
        internal_metrics, external_metrics, knobs
        metric_data : internal metrics
        knobs_data : configuration knobs
        ex. data = {'columnlabels'=array(['metrics_1', 'metrics_2', ...]),
                    'rowlabels'=array([1, 2, ...]),
                    'data'=array([[1,2,3,...], [2,3,4,...], ...[]])}
    '''

    # if opt.workload == 'write':
    #     internal_metrics_path = "../data/redis_data/results/WO_workload/result_internal_total.csv"
    #     external_metrics_path = "../data/redis_data/results/WO_workload/result_external_total.csv"
    # elif opt.workload == 'readwrite':
    #     internal_metrics_path = "../data/redis_data/results/RW_workload/result_internal_total.csv"
    #     external_metrics_path = "../data/redis_data/results/RW_workload/result_external_total.csv"
    # else:
    #     assert False, 'Choose workload write or readwrite'

    internal_metrics_path = os.path.join("./data" ,"result_rdb_internal.csv")
    external_metrics_path = os.path.join("./data" ,"result_rdb_external.csv")
    
    # logger.info("####################Target workload name is {}".format(opt.workload))

    knobs_path = os.path.join(DATA_PATH, "configs")

    if opt.persistence == "RDB":
        knob_data,_ = utils.load_knobs(knobs_path)
    elif opt.persistence == "AOF":
        _,knob_data = utils.load_knobs(knobs_path)
    logger.info("Fin Load Knob_data")

    train_knob_data = {}
    test_knob_data = {}
    train_internal_data = {}
    test_internal_data = {}
    train_external_data = {}
    test_external_data = {}
    
    internal_metric_data, dict_le_in = utils.load_metrics(m_path = internal_metrics_path,
                                                 labels = knob_data['rowlabels'],
                                                 mode = 'internal')

    logger.info("Fin Load internal_metrics_data")

    external_metric_data, _ = utils.load_metrics(m_path = external_metrics_path,
                                                labels = knob_data['rowlabels'],
                                                metrics = ['Totals_Ops/sec'],
                                                mode = 'external')
    logger.info("Fin Load external_metrics_data")
    train_knob_data['data'], test_knob_data['data'] = train_test_split(knob_data['data'],test_size=0.5,shuffle=True,random_state=1004)
    train_knob_data['rowlabels'], test_knob_data['rowlabels'] = train_test_split(knob_data['rowlabels'],test_size=0.5,shuffle=True,random_state=1004)
    train_knob_data['columnlabels'], test_knob_data['columnlabels'] = knob_data['columnlabels'], knob_data['columnlabels']

    train_internal_data['data'], test_internal_data['data'] = train_test_split(internal_metric_data['data'],test_size=0.5,shuffle=True,random_state=1004)
    train_internal_data['rowlabels'], test_internal_data['rowlabels'] = train_test_split(internal_metric_data['rowlabels'],test_size=0.5,shuffle=True,random_state=1004)
    train_internal_data['columnlabels'], test_internal_data['columnlabels'] = internal_metric_data['columnlabels'], internal_metric_data['columnlabels']

    train_external_data['data'], test_external_data['data'] = train_test_split(external_metric_data['data'],test_size=0.5,shuffle=True,random_state=1004)
    train_external_data['rowlabels'], test_external_data['rowlabels'] = train_test_split(external_metric_data['rowlabels'],test_size=0.5,shuffle=True,random_state=1004)
    train_external_data['columnlabels'] = external_metric_data['columnlabels']
    test_external_data['columnlabels'] = external_metric_data['columnlabels']
    assert all(train_knob_data['rowlabels']==train_internal_data['rowlabels'])

    print(f'{opt.persistence} knob data = {len(knob_data)}, {knob_data.keys()}, {knob_data}')
    #print(f'{opt.persistence} metric data = {len(metric_data)}, {metric_data.keys()}, {metric_data}') <- 수정!!
    


    ### METRICS SIMPLIFICATION STAGE ###
    """
        For example,
            pruned_metrics : ['allocator_rss_bytes', 'rss_overhead_bytes', 'used_memory_dataset', 'rdb_last_cow_size']
    """
    logger.info("\n\n====================== run_workload_characterization ====================")
    pruned_metrics = run_workload_characterization(train_internal_data)
    logger.info("Done pruning metrics for workload {} (# of pruned metrics: {}).\n\n""Pruned metrics: {}\n".format(opt.persistence, len(pruned_metrics),pruned_metrics))
    metric_idxs = [i for i, metric_name in enumerate(train_internal_data['columnlabels']) if metric_name in pruned_metrics]
    ranked_metric_data = {
        'data' : train_internal_data['data'][:,metric_idxs],
        'rowlabels' : copy.deepcopy(train_internal_data['rowlabels']),
        'columnlabels' : [train_internal_data['columnlabels'][i] for i in metric_idxs]
    }


    ### KNOBS RANKING STAGE ###
    rank_knob_data = copy.deepcopy(train_knob_data)
    logger.info("\n\n====================== run_knob_identification ====================")
    logger.info("use mode = {}".format(opt.rki))
    ranked_knobs = run_knob_identification(knob_data = rank_knob_data,
                                            metric_data = ranked_metric_data,
                                            mode = opt.rki,
                                            logger = logger)
    logger.info("Done ranking knobs for workload {} (# ranked knobs: {}).\n\n"
                 "Ranked knobs: {}\n".format(opt.persistence, len(ranked_knobs), ranked_knobs))


    ### WORKLOAD MAPPING ###
    ## TODO: ...                 



    ### RECOMMENDATION STAGE ###
    ##TODO: choose k like incremental 4, 8, 16, ...
    top_ks = [5, 10, 15, 19]
    best_recommend = -float('inf')
    best_topk = None
    best_conf_map = None
    for top_k in top_ks:        
        logger.info("\n\n================ The number of TOP knobs ===============")
        logger.info(top_k)

        ranked_test_knob_data = utils.get_ranked_knob_data(ranked_knobs, test_knob_data, top_k)
        
        ## TODO: params(GP option) and will offer opt all
        FIN,recommend,conf_map = configuration_recommendation(ranked_test_knob_data,test_external_data, logger, opt.gp, opt.db, opt.persistence)

        # if recommend > best_recommend and FIN:
        #     best_recommend = recommend
        #     best_topk = top_k
        #     best_conf_map = conf_map
        # logger.info("Best top_k")
        # logger.info(best_topk)
        # print(best_topk)

        ## Generate Best Configuration file for Redis
        print(opt.persistence)
        utils.convert_dict_to_conf(conf_map, opt.persistence)

    print("END TRAIN")