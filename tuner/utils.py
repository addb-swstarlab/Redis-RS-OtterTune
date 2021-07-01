# -*- coding: utf-8 -*-

import json
import numpy as np

import os
import logging
import logging.handlers

import datetime

import torch

def get_logger(log_path='./logs'):

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = logging.getLogger()
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s %(message)s', date_format)
    i = 0
    today = datetime.datetime.now()
    name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    while os.path.exists(os.path.join(log_path, name)):
        i += 1
        name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    
    fileHandler = logging.FileHandler(os.path.join(log_path, name))
    streamHandler = logging.StreamHandler()
    
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    logger.setLevel(logging.INFO)
    logger.info('Writing logs at {}'.format(os.path.join(log_path, name)))
    return logger, os.path.join(log_path, name)


def make_date_dir(path):
    """
    :param path
    :return: os.path.join(path, date_dir)
    """
    if not os.path.exists(path):
        os.mkdir(path)
    i = 0
    today = datetime.datetime.now()
    name = today.strftime('%Y%m%d')+'-'+'%02d' % i

    while os.path.exists(os.path.join(path, name)):
        i += 1
        name = today.strftime('%Y%m%d')+'-'+'%02d' % i
        
    os.mkdir(os.path.join(path, name))
    return os.path.join(path, name)


def get_ranked_knob_data(ranked_knobs, knob_data, top_k):
    '''
        ranked_knobs: sorted knobs with ranking 
                        ex. ['m3', 'm6', 'm2', ...]
        knob_data: dictionary data with keys(columnlabels, rowlabels, data)
        top_k: A standard to split knobs 
    '''
    ranked_knob_data = knob_data.copy()
    ranked_knob_data['columnlabels'] = np.array(ranked_knobs)
        
    for i, knob in enumerate(ranked_knobs):
        ranked_knob_data['data'][:,i] = knob_data['data'][:, list(knob_data['columnlabels']).index(knob)]
    
    # pruning with top_k
    ranked_knob_data['data'] = ranked_knob_data['data'][:,:top_k]
    ranked_knob_data['columnlabels'] = ranked_knob_data['columnlabels'][:top_k]

    #print('pruning data with ranking')
    #print('Pruned Ranked knobs: ', ranked_knob_data['columnlabels'])

    return ranked_knob_data

def collate_function(examples):
    knobs=[None]*len(examples)
    EMs=[None]*len(examples)
    for i,(knob,EM) in enumerate(examples):
        knobs[i] = knob
        EMs[i] = EM
    return torch.tensor(knobs),torch.tensor(EMs)

def convert_dict_to_conf(rec_config, persistence):
    f = open('../data/redis_data/init_config.conf', 'r')
    json_configs_path = '../data/redis_data/'+persistence+'_knobs.json'
    with open(json_configs_path, 'r') as j:
        json_configs = json.load(j)

    dict_config = {}
    for d in json_configs:
        dict_config[d['name']] = d['default']

    config_list = f.readlines()
    save_f = False
    categorical_knobs = ['appendonly', 'no-appendfsync-on-rewrite', 'aof-rewrite-incremental-fsync',
                         'aof-use-rdb-preamble', 'rdbcompression', 'rdbchecksum', 
                         'rdb-save-incremental-fsync', 'activedefrag', 'activerehashing']
    
    if persistence == "RDB":
        save_sec = []
        save_changes = []   
    
    
    for k in dict_config.keys():
        if k in rec_config.keys():
            dict_config[k] = rec_config[k]

        dict_config[k] = round(dict_config[k])
        
        if k in categorical_knobs:
            if k == "activerehashing":
                if dict_config[k] == 0: dict_config[k] = 'no'
                elif dict_config[k] >= 1 : dict_config[k] = 'yes'
            else:
                if dict_config[k] == 0: dict_config[k] = 'no'
                elif dict_config[k] == 1: dict_config[k] = 'yes'
        if k == 'appendfsync':
            if dict_config[k] == 0: dict_config[k] = 'always'
            elif dict_config[k] == 1: dict_config[k] = 'everysec'
            elif dict_config[k] >= 2: dict_config[k] = 'no'    

        if 'changes' in k or 'sec' in k:
            save_f = True
            if 'sec' in k:
                save_sec.append(dict_config[k])
            if 'changes' in k:
                save_changes.append(dict_config[k])
            continue
        
        if k == 'auto-aof-rewrite-min-size':
            dict_config[k] = str(dict_config[k]) + 'mb'

        config_list.append(k+' '+str(dict_config[k])+'\n')
    
    if save_f:
        for s in range(len(save_sec)):
            config_list.append('save ' + str(save_sec[s]) + ' ' + str(save_changes[s]) + '\n')
    i = 0
    PATH = '../data/redis_data/config_results/{}'.format(persistence)
    NAME = persistence+'_rec_config{}.conf'.format(i)
    while os.path.exists(os.path.join(PATH,NAME)):
        i+=1
        NAME = persistence+'_rec_config{}.conf'.format(i)
    
    with open(os.path.join(PATH,NAME), 'w') as rec_f:
        rec_f.writelines(config_list) 

def config_exist(persistence):
    i = 0
    PATH = '../data/redis_data/config_results/{}'.format(persistence)
    NAME = persistence+'_rec_config{}.conf'.format(i)
    while os.path.exists(os.path.join(PATH,NAME)):
        i+=1
        NAME = persistence+'_rec_config{}.conf'.format(i)
    return NAME[:-5]



from sklearn.preprocessing import StandardScaler

# Modifying to import upper folder
import sys
sys.path.append('../')


def process_training_data(target_knob, target_metric, db_type, data_type):
    # Load mapped workload data
    # TODO: If we have mapped_workload, we will use this code
    
    # if target_data['mapped_workload'] is not None:
        # mapped_workload_id = target_data['mapped_workload'][0]
        # mapped_workload = Workload.objects.get(pk=mapped_workload_id)
        # workload_knob_data = PipelineData.objects.get(
        #     pipeline_run=latest_pipeline_run,
        #     workload=mapped_workload,
        #     task_type=PipelineTaskType.KNOB_DATA)
        # workload_knob_data = JSONUtil.loads(workload_knob_data.data)
        # workload_metric_data = PipelineData.objects.get(
        #     pipeline_run=latest_pipeline_run,
        #     workload=mapped_workload,
        #     task_type=PipelineTaskType.METRIC_DATA)
        # workload_metric_data = JSONUtil.loads(workload_metric_data.data)
        # cleaned_workload_knob_data = DataUtil.clean_knob_data(workload_knob_data["data"],
        #                                                       workload_knob_data["columnlabels"],
        #                                                       [newest_result.session])
        # X_workload = np.array(cleaned_workload_knob_data[0])
        # X_columnlabels = np.array(cleaned_workload_knob_data[1])
        # y_workload = np.array(workload_metric_data['data'])
        # y_columnlabels = np.array(workload_metric_data['columnlabels'])
        # rowlabels_workload = np.array(workload_metric_data['rowlabels'])
    if False:
        pass
    else:
        # combine the target_data with itself is actually adding nothing to the target_data
        X_workload = np.array(target_knob['data'])
        X_columnlabels = np.array(target_knob['columnlabels'])
        y_workload = np.array(target_metric['data'])
        y_columnlabels = np.array(target_metric['columnlabels'])
        rowlabels_workload = np.array(target_knob['rowlabels'])

    # Target workload data
    X_target = target_knob['data']
    y_target = target_metric['data']
    rowlabels_target = np.array(target_knob['rowlabels'])

    if not np.array_equal(X_columnlabels, target_knob['columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical X columnlabels (sorted knob names)'),
                        X_columnlabels, target_knob['X_columnlabels'])
    if not np.array_equal(y_columnlabels, target_metric['columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical y columnlabels (sorted metric names)'),
                        y_columnlabels, target_metric['columnlabels'])

    # TODO: If we have mapped_workload, we will use this code
    # Filter ys by current target objective metric
    # target_objective = newest_result.session.target_objective
    # target_obj_idx = [i for i, cl in enumerate(y_columnlabels) if cl == target_objective]
    # if len(target_obj_idx) == 0:
    #     raise Exception(('Could not find target objective in metrics '
    #                      '(target_obj={})').format(target_objective))
    # elif len(target_obj_idx) > 1:
    #     raise Exception(('Found {} instances of target objective in '
    #                      'metrics (target_obj={})').format(len(target_obj_idx),
    #                                                        target_objective))

    # y_workload = y_workload[:, target_obj_idx]
    # y_target = y_target[:, target_obj_idx]
    # y_columnlabels = y_columnlabels[target_obj_idx]

    # y_workload = y_workload[:, 0]
    # y_target = y_target[:, 0]
    # y_columnlabels = y_columnlabels[0]

    # Combine duplicate rows in the target/workload data (separately)
    X_workload, y_workload, rowlabels_workload = DataUtil.combine_duplicate_rows(
        X_workload, y_workload, rowlabels_workload)
    X_target, y_target, rowlabels_target = DataUtil.combine_duplicate_rows(
        X_target, y_target, rowlabels_target)

    # Delete any rows that appear in both the workload data and the target
    # data from the workload data
    # dups_filter = np.ones(X_workload.shape[0], dtype=bool)
    # target_row_tups = [tuple(row) for row in X_target]
    # for i, row in enumerate(X_workload):
    #     if tuple(row) in target_row_tups:
    #         dups_filter[i] = False
    # X_workload = X_workload[dups_filter, :]
    # y_workload = y_workload[dups_filter, :]
    # rowlabels_workload = rowlabels_workload[dups_filter]

    # Combine target & workload Xs for preprocessing
    X_matrix = np.vstack([X_target,X_workload])

    dummy_encoder = None
    
    # Scale to N(0, 1)
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_matrix)
    if y_target.shape[0] < 5:  # FIXME
        # FIXME (dva): if there are fewer than 5 target results so far
        # then scale the y values (metrics) using the workload's
        # y_scaler. I'm not sure if 5 is the right cutoff.
        y_target_scaler = None
        y_workload_scaler = StandardScaler()
        y_matrix = np.vstack([y_target, y_workload])
        y_scaled = y_workload_scaler.fit_transform(y_matrix)
    else:
        # FIXME (dva): otherwise try to compute a separate y_scaler for
        # the target and scale them separately.
        try:
            y_target_scaler = StandardScaler()
            y_workload_scaler = StandardScaler()
            y_target_scaled = y_target_scaler.fit_transform(y_target)
            y_workload_scaled = y_workload_scaler.fit_transform(y_workload)
            y_scaled = np.vstack([y_target_scaled, y_workload_scaled])
        except ValueError:
            y_target_scaler = None
            y_workload_scaler = StandardScaler()
            y_scaled = y_workload_scaler.fit_transform(y_target)

    # Maximize the throughput, moreisbetter
    # If Use gradient descent to minimize -throughput
    # if not lessisbetter:
    #     y_scaled = -y_scaled

    # FIXME (dva): check if these are good values for the ridge
    # ridge = np.empty(X_scaled.shape[0])
    # ridge[:X_target.shape[0]] = 0.01
    # ridge[X_target.shape[0]:] = 0.1
    X_min = np.empty(X_scaled.shape[1])
    X_max = np.empty(X_scaled.shape[1])
    X_scaler_matrix = np.zeros([1, X_scaled.shape[1]])

    with open(os.path.join("../data/{}_data".format(db_type),data_type+"_knobs.json"), "r") as data:
        session_knobs = json.load(data)

    # Set min/max for knob values
    #TODO : we make binary_index_set
    for i in range(X_scaled.shape[1]):
        col_min = X_scaled[:, i].min()
        col_max = X_scaled[:, i].max()
        for knob in session_knobs:
            if X_columnlabels[i] == knob["name"]:
                if knob["minval"]==0:
                    col_min = knob["minval"]
                    col_max = knob["maxval"]
                else:
                    X_scaler_matrix[0][i] = knob["minval"]
                    col_min = X_scaler.transform(X_scaler_matrix)[0][i]
                    X_scaler_matrix[0][i] = knob["maxval"]
                    col_max = X_scaler.transform(X_scaler_matrix)[0][i]
            X_min[i] = col_min
            X_max[i] = col_max

    return X_columnlabels, X_scaler, X_scaled, y_scaled, X_max, X_min, dummy_encoder