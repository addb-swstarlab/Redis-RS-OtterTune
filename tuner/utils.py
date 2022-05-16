# -*- coding: utf-8 -*-

import time, os, sys
import pickle, json
import logging
import datetime
import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
import logging


def time_start():
    return time.time()


def time_end(start):
    end = time.time()
    delay = end - start
    return delay


def get_timestamp():
    """
    获取UNIX时间戳
    """
    return int(time.time())


def time_to_str(timestamp):
    """
    将时间戳转换成[YYYY-MM-DD HH:mm:ss]格式
    """
    return datetime.datetime. \
        fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


class Logger:

    def __init__(self, name, log_file=''):
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        self.logger.addHandler(sh)
        if len(log_file) > 0:
            self.log2file = True
        else:
            self.log2file = False

    def _write_file(self, msg):
        pass
        # if self.log2file:
        #     with open(self.log_file, 'a+') as f:
        #         f.write(msg + '\n')

    def get_timestr(self):
        timestamp = get_timestamp()
        date_str = time_to_str(timestamp)
        return date_str

    def warn(self, msg):
        msg = "%s[WARN] %s" % (self.get_timestr(), msg)
        self.logger.warning(msg)
        self._write_file(msg)

    def info(self, msg):
        msg = "%s[INFO] %s" % (self.get_timestr(), msg)
        self.logger.info(msg)
        self._write_file(msg)

    def error(self, msg):
        msg = "%s[ERROR] %s" % (self.get_timestr(), msg)
        self.logger.error(msg)
        self._write_file(msg)


def save_state_actions(state_action, filename):
    f = open(filename, 'wb')
    pickle.dump(state_action, f)
    f.close()


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
    RDB_datas = []
    RDB_columns = []
    RDB_rowlabels = []
    AOF_datas = []
    AOF_columns = []
    AOF_rowlabels = []
    ISAOF = 0
    ISRDB = 1

    config_nums = [_ for _ in range(1, 201)] + [_ for _ in range(10001, 10201)]

    for m in config_nums:
        flag = 0
        datas = []
        columns = []
        knob_path = os.path.join(knobs_path, 'config' + str(m) + '.conf')
        f = open(knob_path, 'r')
        config_file = f.readlines()
        knobs_list = config_file[config_file.index('#rdb-save-incremental-fsync yes\n') + 1:]

        cnt = 1

        for l in knobs_list:
            if l.split()[0] != 'save':
                col, d = l.split(' ', 1)
                d = d.replace('\n', '')
                if d.isalpha():
                    if d in ["no", "yes"]:
                        d = ["no", "yes"].index(d)
                    elif d in ["always", "everysec", "no", "noeviction"]:
                        d = ["always", "everysec", "no", "noeviction"].index(d)
                elif d.endswith("mb"):
                    d = d[:-2]
                elif d.endswith("gb"):
                    d = float(d[:-2]) * 1000

                if d in ["volatile-lfu", "volatile-random", "volatile-lru", "volatile-ttl", "allkeys-lru",
                         "allkeys-lfu", "allkeys-random"]:
                    d = ["volatile-lfu", "volatile-random", "volatile-lru", "volatile-ttl", "allkeys-lru",
                         "allkeys-lfu", "allkeys-random"].index(d)
                datas.append(d)
                columns.append(col)
            else:
                col, d1, d2 = l.split()
                columns.append(col + str(cnt) + "_sec")
                columns.append(col + str(cnt) + "_changes")
                datas.append(d1)
                datas.append(d2)
                cnt += 1

            if l.split()[0] == 'appendonly':
                flag = ISAOF
            if l.split()[0] == 'save':
                flag = ISRDB

        # add active knobs
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
        datas = list(map(int, datas))
        if flag == ISRDB:
            #         print('RDB')
            RDB_datas.append(datas)
            RDB_columns.append(columns)
            RDB_rowlabels.append(m + 1)
        if flag == ISAOF:
            #         print('AOF')
            AOF_datas.append(datas)
            AOF_columns.append(columns)
            AOF_rowlabels.append(m + 1)

    dict_RDB['data'] = np.array(RDB_datas)
    dict_RDB['rowlabels'] = np.array(RDB_rowlabels)
    dict_RDB['columnlabels'] = np.array(RDB_columns[0])
    dict_AOF['data'] = np.array(AOF_datas)
    dict_AOF['rowlabels'] = np.array(AOF_rowlabels)
    dict_AOF['columnlabels'] = np.array(AOF_columns[0])
    return dict_RDB, dict_AOF


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
    if labels[0] < 10000:
        tmp_rowlabels = [_ - 2 for _ in labels]
    else:
        tmp_rowlabels = [_ - 10002 for _ in labels]

    pd_metrics = pd_metrics.iloc[tmp_rowlabels][:]
    nan_columns = pd_metrics.columns[pd_metrics.isnull().any()]
    pd_metrics = pd_metrics.drop(columns=nan_columns)

    # for i in range(len(pd_metrics)):
    #     columns.append(pd_metrics.columns.to_list())
    dict_metrics['columnlabels'] = np.array(pd_metrics.columns)
    # dict_metrics['columnlabels'] = np.array(itemgetter(*tmp_rowlabels)(dict_metrics['columnlabels'].tolist()))
    dict_metrics['rowlabels'] = np.array(labels)
    dict_metrics['data'] = np.array(pd_metrics.values)

    return dict_metrics


def load_metrics(m_path=' ', labels=[], metrics=None, mode=' '):
    if mode == "internal":
        pd_metrics = pd.read_csv(m_path)
        pd_metrics, dict_le = metric_preprocess(pd_metrics)
        return metrics_make_dict(pd_metrics, labels), dict_le
    else:
        pd_metrics = pd.read_csv(m_path)
        # pd_metrics, dict_le = metric_preprocess(pd_metrics)
        return metrics_make_dict(pd_metrics[metrics], labels), None


def load_knobs(k_path):
    return knobs_make_dict(k_path)


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
        ranked_knob_data['data'][:, i] = knob_data['data'][:, list(knob_data['columnlabels']).index(knob)]

    # pruning with top_k
    ranked_knob_data['data'] = ranked_knob_data['data'][:, :top_k]
    ranked_knob_data['columnlabels'] = ranked_knob_data['columnlabels'][:top_k]

    # print('pruning data with ranking')
    # print('Pruned Ranked knobs: ', ranked_knob_data['columnlabels'])

    return ranked_knob_data


def convert_dict_to_conf(rec_config, persistence):
    f = open('../data/init_config.conf', 'r')
    json_configs_path = '../data/' + persistence + '_knobs.json'
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

    # if persistence == "RDB":
    #     save_sec = []
    #     save_changes = []
    save_sec = []
    save_changes = []

    for k in dict_config.keys():
        if k in rec_config.keys():
            dict_config[k] = rec_config[k]

        dict_config[k] = round(dict_config[k])

        if k in categorical_knobs:
            if k == "activerehashing":
                if dict_config[k] == 0:
                    dict_config[k] = 'no'
                elif dict_config[k] >= 1:
                    dict_config[k] = 'yes'
            else:
                if dict_config[k] == 0:
                    dict_config[k] = 'no'
                elif dict_config[k] >= 1:
                    dict_config[k] = 'yes'
        if k == 'appendfsync':
            if dict_config[k] == 0:
                dict_config[k] = 'always'
            elif dict_config[k] == 1:
                dict_config[k] = 'everysec'
            elif dict_config[k] >= 2:
                dict_config[k] = 'no'

        if 'changes' in k or 'sec' in k:
            save_f = True
            if 'sec' in k:
                save_sec.append(dict_config[k])
            if 'changes' in k:
                save_changes.append(dict_config[k])
            continue

        if k == 'auto-aof-rewrite-min-size':
            dict_config[k] = str(dict_config[k]) + 'mb'

        config_list.append(k + ' ' + str(dict_config[k]) + '\n')

    if save_f:
        for s in range(len(save_sec)):
            config_list.append('save ' + str(save_sec[s]) + ' ' + str(save_changes[s]) + '\n')
    i = 0
    PATH = '../data/config_results/{}'.format(persistence)
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    NAME = persistence + '_rec_config{}.conf'.format(i)
    while os.path.exists(os.path.join(PATH, NAME)):
        i += 1
        NAME = persistence + '_rec_config{}.conf'.format(i)

    with open(os.path.join(PATH, NAME), 'w') as rec_f:
        rec_f.writelines(config_list)


def config_exist(persistence):
    i = 0
    PATH = '../data/config_results/{}'.format(persistence)
    NAME = persistence + '_rec_config{}.conf'.format(i)
    while os.path.exists(os.path.join(PATH, NAME)):
        i += 1
        NAME = persistence + '_rec_config{}.conf'.format(i)
    return NAME[:-5]


from sklearn.preprocessing import StandardScaler

sys.path.append('../')
from models.util import DataUtil


def process_training_data(target_knob, target_metric):
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
    X_matrix = np.vstack([X_target, X_workload])

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

    with open(os.path.join("../data/RDB_knobs.json"), "r") as data:
        session_knobs = json.load(data)

    # Set min/max for knob values
    # TODO : we make binary_index_set
    for i in range(X_scaled.shape[1]):
        col_min = X_scaled[:, i].min()
        col_max = X_scaled[:, i].max()
        for knob in session_knobs:
            if X_columnlabels[i] == knob["name"]:
                if knob["minval"] == 0:
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
