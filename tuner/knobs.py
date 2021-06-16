import os

from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd

def knobs_make_dict(knobs_path):
    '''
        input: DataFrame form (samples_num, knobs_num)
        output: Dictionary form --> RDB and AOF
            ex. dict_knobs = {'columnlabels'=array([['knobs_1', 'knobs_2', ...],['knobs_1', 'knobs_2', ...], ...]),
                                'rowlabels'=array([1, 2, ...]),
                                'data'=array([[1,2,3,...], [2,3,4,...], ...[]])}

        For mode selection knob, "yes" -> 1 , "no" -> 0
    '''
    config_files = os.listdir(knobs_path)

    dict_RDB = {}
    dict_AOF = {}
    RDB_datas, RDB_columns, RDB_rowlabels = [], [], []
    AOF_datas, AOF_columns, AOF_rowlabels = [], [], []
    ISAOF = 0
    ISRDB = 1

    for m in range(len(config_files)):
        flag = 0
        datas, columns = [], []
        knob_path = os.path.join(knobs_path, 'config'+str(m+1)+'.conf')
        f = open(knob_path, 'r')
        config_file = f.readlines()
        knobs_list = config_file[62:]
        #knobs_list = config_file[config_file.index('\n')+1:]
        cnt = 1

        for knobs in knobs_list:
            if knobs.split()[0] != 'save':
                knob, data = knobs.strip().split()
                if data.isalpha() or '-' in data:
                    if data in ["no","yes"]:
                        data = ["no","yes"].index(data)
                    elif data in ["always","everysec","no"]:
                        data = ["always","everysec","no"].index(data)
                    elif data in ["volatile-lru","allkeys-lru","volatile-lfu","allkeys-lfu","volatile-random","allkeys-random","volatile-ttl","noeviction"]:
                        data = ["volatile-lru","allkeys-lru","volatile-lfu","allkeys-lfu","volatile-random","allkeys-random","volatile-ttl","noeviction"].index(data)
                elif data.endswith("mb") or data.endswith("gb"):
                    data = data[:-2]
                datas.append(data)
                columns.append(knob)
            else:
                knob, data1, data2 = knobs.split()
                columns.append(knob+str(cnt)+"_sec")
                columns.append(knob+str(cnt)+"_changes")
                datas.append(data1)
                datas.append(data2)
                cnt += 1

            if knobs.split()[0] == 'appendonly':
                flag = ISAOF
            if knobs.split()[0] == 'save':
                flag = ISRDB

        # add active knobs when activedefrag is on annotation.
        if "activedefrag" not in columns:
            columns.append("activedefrag")
            # "0" means no
            datas.append("0")
            columns.append("active-defrag-threshold-lower")
            datas.append(10)
            columns.append("active-defrag-threshold-upper")
            datas.append(100)
            columns.append("active-defrag-cycle-min")
            datas.append(5)
            columns.append("active-defrag-cycle-max")
            datas.append(75)

        def str2Numbers(str):
            try:
                number = int(str)
            except:
                number = float(str)
            return number

        datas = list(map(str2Numbers,datas))
        if flag == ISRDB:
    #         print('RDB')
            RDB_datas.append(datas)
            RDB_columns.append(columns)
            RDB_rowlabels.append(m+1-10000)
        if flag == ISAOF: 
    #         print('AOF')
            AOF_datas.append(datas)
            AOF_columns.append(columns)
            AOF_rowlabels.append(m+1)

    dict_RDB['data'] = np.array(RDB_datas)
    dict_RDB['rowlabels'] = np.array(RDB_rowlabels)
    dict_RDB['columnlabels'] = np.array(RDB_columns[0])
    dict_AOF['data'] = np.array(AOF_datas)
    dict_AOF['rowlabels'] = np.array(AOF_rowlabels)
    dict_AOF['columnlabels'] = np.array(AOF_columns[0])
    return dict_RDB, dict_AOF


def aggregateInternalMetrics(internal_metric_datas):
    aggregated_IM_data = {}
    for workload in internal_metric_datas.keys():
        if workload.startswith('workload'):
            if not (aggregated_IM_data.get('data') is None):
                aggregated_IM_data['data'] = np.concatenate((aggregated_IM_data['data'],internal_metric_datas[workload]))
            else:
                aggregated_IM_data['data'] = internal_metric_datas[workload]
        else:
            aggregated_IM_data[workload] = internal_metric_datas[workload]
    return aggregated_IM_data


def metric_preprocess(metrics):
    '''To invert for categorical internal metrics'''
    dict_le = {}
    c_metrics = metrics.copy()

    for col in metrics.columns:
        if isinstance(c_metrics[col][0], str):
            le = LabelEncoder()
            c_metrics[col] = le.fit_transform(c_metrics[col])
            dict_le[col] = le
    return c_metrics, dict_le

def metrics_make_dict(pd_metrics, labels):
    '''
        input: DataFrame form (samples_num, metrics_num)
        output: Dictionary form
            ex. dict_metrics = {'columnlabels'=array([['metrics_1', 'metrics_2', ...],['metrics_1', 'metrics_2', ...], ...]),
                            'rowlabels'=array([1, 2, ...]),
                            'data'=array([[1,2,3,...], [2,3,4,...], ...[]])}
    '''
    # labels = RDB or AOF rowlabels
    
    dict_metrics = {}
    tmp_rowlabels = [_-1 for _ in labels]
    pd_metrics = pd_metrics.iloc[tmp_rowlabels][:]
    nan_columns = pd_metrics.columns[pd_metrics.isnull().any()]
    pd_metrics = pd_metrics.drop(columns=nan_columns)
    # for i in range(len(pd_metrics)):
    #     columns.append(pd_metrics.columns.to_list())
    dict_metrics['columnlabels'] = np.array(pd_metrics.columns)
    #dict_metrics['columnlabels'] = np.array(itemgetter(*tmp_rowlabels)(dict_metrics['columnlabels'].tolist()))
    dict_metrics['rowlabels'] = np.array(labels)
    dict_metrics['data'] = np.array(pd_metrics.values)
    
    return dict_metrics    
    

def load_metrics(metric_path = ' ', labels = [], metrics=None):
    """ 
    If metrics is None, it means internal metrics.
    We may change external because of using two metric.
    """
    if metrics is None:
        pd_metrics = pd.read_csv(metric_path)
        pd_metrics, dict_le = metric_preprocess(pd_metrics)
        return metrics_make_dict(pd_metrics, labels), dict_le
    else:
        pd_metrics = pd.read_csv(metric_path)
        return metrics_make_dict(pd_metrics[metrics], labels), None

def load_knobs(knobs_path):
    return knobs_make_dict(knobs_path)