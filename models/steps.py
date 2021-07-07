import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.cluster import GapStatistic, KMeansClusters, create_kselection_model, MeanShiftClustering, GMMClustering
from models.factor_analysis import FactorAnalysis
from models.preprocessing import (get_shuffle_indices, consolidate_columnlabels)
from models.redisDataset import RedisDataset
from models.ranking import Ranking
from models.dnn import RedisSingleDNN, RedisTwiceDNN

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader,RandomSampler

import utils
import knobs

DATA_PATH = "../data/redis_data"
DEVICE = torch.device("cpu")

import warnings

warnings.filterwarnings('ignore')

def dataPreprocessing(target_num, persistence,logger):
    target_DATA_PATH = "../data/redis_data/workload{}".format(target_num)
    
    knobs_path = os.path.join(DATA_PATH, "configs")
    if persistence == "RDB":
        knob_data, _ = knobs.load_knobs(knobs_path)
    elif persistence == "AOF":
        _, knob_data = knobs.load_knobs(knobs_path)

    logger.info("Finish Load Knob Data")

    internal_metric_datas = {}
    external_metric_datas = {}

    # len()-1 because of configs dir
    for i in range(1,len(os.listdir(DATA_PATH))):
    #for i in range(1,10):
        if target_num == i:
            target_external_data, _ = knobs.load_metrics(metric_path = os.path.join(target_DATA_PATH ,f"result_{persistence.lower()}_external_{i}.csv"),
                                                labels = knob_data['rowlabels'],
                                                metrics = ['Totals_Ops/sec', 'Totals_p99_Latency'])
        else:
            internal_metric_data, _ = knobs.load_metrics(metric_path = os.path.join(DATA_PATH,f'workload{i}',f'result_{persistence.lower()}_internal_{i}.csv'),
                                                            labels = knob_data['rowlabels'])
            
            external_metric_data, _ = knobs.load_metrics(metric_path = os.path.join(DATA_PATH,f'workload{i}',f'result_{persistence.lower()}_external_{i}.csv'),
                                                labels = knob_data['rowlabels'],
                                                metrics = ['Totals_Ops/sec', 'Totals_p99_Latency'])
            internal_metric_datas[f'workload{i}'] = internal_metric_data['data']
            external_metric_datas[f'workload{i}'] = external_metric_data['data']

    internal_metric_datas['columnlabels'] = internal_metric_data['columnlabels']
    internal_metric_datas['rowlabels'] = internal_metric_data['rowlabels']
    external_metric_datas['columnlabels'] = ['Totals_Ops/sec', 'Totals_p99_Latency']
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

    aggregated_IM_data = knobs.aggregateMetrics(internal_metric_datas)
    aggregated_EM_data = knobs.aggregateMetrics(external_metric_datas)

    """
    data = concat((workload2,...,workload18)) length = 10000 * N of workload
    columnlabels  = same as internal_metric_datas's columnlabels
    rowlabels = same as internal_metric_datas's rowlabels

    aggregated_IM_data = {
        'data'=array([[1,2,3,...], [2,3,4,...], ...[]])
        'columnlabels'=array(['IM_1', 'IM_2', ...]),
        'rowlabels'=array([1, 2, ..., 10000])}
    
    """
    return knob_data, aggregated_IM_data, aggregated_EM_data, target_external_data

#Step 1
def metricSimplification(metric_data, logger, args):
    matrix = metric_data['data']
    columnlabels = metric_data['columnlabels']

    # Remove any constant columns
    nonconst_matrix = []
    nonconst_columnlabels = []
    for col, (_,v) in zip(matrix.T, enumerate(columnlabels)):
        if np.any(col != col[0]):
            nonconst_matrix.append(col.reshape(-1, 1))
            nonconst_columnlabels.append(v)
    assert len(nonconst_matrix) > 0, "Need more data to train the model"
    nonconst_matrix = np.hstack(nonconst_matrix)
    logger.info("Workload characterization ~ nonconst data size: %s", nonconst_matrix.shape)

    # Remove any duplicate columns
    unique_matrix, unique_idxs = np.unique(nonconst_matrix, axis=1, return_index=True)
    unique_columnlabels = [nonconst_columnlabels[idx] for idx in unique_idxs]

    logger.info("Workload characterization ~ final data size: %s", unique_matrix.shape)
    n_rows, n_cols = unique_matrix.shape

    # Shuffle the matrix rows
    shuffle_indices = get_shuffle_indices(n_rows)
    shuffled_matrix = unique_matrix[shuffle_indices, :]

    shuffled_matrix = MinMaxScaler().fit_transform(shuffled_matrix)
    #shuffled_matrix = StandardScaler().fit_transform(shuffled_matrix)

    #FactorAnalysis
    fa_model = FactorAnalysis()
    fa_model.fit(shuffled_matrix, unique_columnlabels, n_components=5)
    # Components: metrics * factors
    components = fa_model.components_.T.copy()

    # Clustering method : Gaussian Mixture Model(GMM)
    logger.info("Clustering mode : {}".format(args.cluster))
    if args.cluster == 'gmm':
        cluster = GMMClustering(components)
        cluster.fit(components)
        pruned_metrics = cluster.get_closest_samples(unique_columnlabels)
        logger.info("Found optimal number of clusters: {}".format(cluster.optimK))
    elif args.cluster == 'k-means':
        #KMeansClusters()
        kmeans_models = KMeansClusters()
        ##TODO: Check Those Options
        kmeans_models.fit(components, min_cluster=1,
                          max_cluster=min(n_cols - 1, 20),
                          sample_labels=unique_columnlabels,
                          estimator_params={'n_init': 100})
        gapk = create_kselection_model("gap-statistic")
        gapk.fit(components, kmeans_models.cluster_map_)

        logger.info("Found optimal number of clusters: {}".format(gapk.optimal_num_clusters_))

        # Get pruned metrics, cloest samples of each cluster center
        pruned_metrics = kmeans_models.cluster_map_[gapk.optimal_num_clusters_].get_closest_samples()
    
    # Clustering method : Mean Shift
    elif args.cluster == 'ms':
        ms = MeanShiftClustering(components)
        ms.fit(components)
        pruned_metrics = ms.get_closest_samples(unique_columnlabels)
        logger.info("Found optimal number of clusters: {}".format(len(ms.centroid)))

    return pruned_metrics


def knobsRanking(knob_data, metric_data, mode, logger):
    """
    knob_data : will be ranked by knobs_ranking
    metric_data : pruned metric_data by metric simplification
    mode : selct knob_identification(like lasso, xgb, rf)
    logger
    """
    knob_matrix = knob_data['data']
    knob_columnlabels = knob_data['columnlabels']

    metric_matrix = metric_data['data']
    #metric_columnlabels = metric_data['columnlabels']

    encoded_knob_columnlabels = knob_columnlabels
    encoded_knob_matrix = knob_matrix

    # standardize values in each column to N(0, 1)
    standardizer = StandardScaler()
    standardized_knob_matrix = standardizer.fit_transform(encoded_knob_matrix)
    standardized_metric_matrix = standardizer.fit_transform(metric_matrix)

    # shuffle rows (note: same shuffle applied to both knob and metric matrices)
    shuffle_indices = get_shuffle_indices(standardized_knob_matrix.shape[0], seed=17)
    shuffled_knob_matrix = standardized_knob_matrix[shuffle_indices, :]
    shuffled_metric_matrix = standardized_metric_matrix[shuffle_indices, :]

    model = Ranking(mode)
    model.fit(shuffled_knob_matrix,shuffled_metric_matrix,encoded_knob_columnlabels)
    encoded_knobs = model.get_ranked_features()
    feature_imp = model.get_ranked_importance()
    if feature_imp is None:
        pass
    else:
        logger.info('Feature importance')
        logger.info(feature_imp)

    consolidated_knobs = consolidate_columnlabels(encoded_knobs)

    return consolidated_knobs


def prepareForTraining(opt, top_k_knobs, aggregated_EM_data, target_external_data):
    with open("../data/workloads_info.json",'r') as f:
        workload_info = json.load(f)

    workloads=np.array([])
    target_workload = np.array([])
    for workload in range(1,len(workload_info.keys())):
        if workload != opt.target:
            if len(workloads) == 0:
                workloads = np.array(workload_info[str(workload)])
            else:
                workloads = np.vstack((workloads,np.array(workload_info[str(workload)])))
        else:
            target_workload = np.array(workload_info[str(workload)])
    
    top_k_knobs = pd.DataFrame(top_k_knobs['data'], columns = top_k_knobs['columnlabels'])
    aggregated_EM_data = pd.DataFrame(aggregated_EM_data['data'], columns = ['Totals_Ops/sec', 'Totals_p99_Latency'])
    workload_infos = pd.DataFrame(workloads,columns = workload_info['info'])
    target_workload = pd.DataFrame([target_workload],columns= workload_info['info'])
    target_external_data = pd.DataFrame(target_external_data['data'], columns = ['Totals_Ops/sec', 'Totals_p99_Latency'])

    top_k_knobs['tmp'] = 1
    workload_infos['tmp'] = 1
    target_workload['tmp'] = 1
    knobWithworkload = pd.merge(top_k_knobs,workload_infos,on=['tmp'])
    knobWithworkload = knobWithworkload.drop('tmp',axis=1)
    targetWorkload = pd.merge(top_k_knobs,target_workload,on=['tmp'])
    targetWorkload = targetWorkload.drop('tmp',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(knobWithworkload, aggregated_EM_data, test_size = 0.33, random_state=42)

    scaler_X = MinMaxScaler().fit(X_train)
    scaler_y = RobustScaler().fit(y_train)

    X_tr = scaler_X.transform(X_train).astype(np.float32)
    X_val = scaler_X.transform(X_val).astype(np.float32)
    y_tr = scaler_y.transform(y_train).astype(np.float32)
    y_val = scaler_y.transform(y_val).astype(np.float32)

    X_te = scaler_X.transform(targetWorkload).astype(np.float32)
    y_te = scaler_y.transform(target_external_data).astype(np.float32)    

    trainDataset = RedisDataset(X_tr, y_tr)
    valDataset = RedisDataset(X_val, y_val)
    testDataset = RedisDataset(X_te, y_te)

    trainSampler = RandomSampler(trainDataset)
    valSampler = RandomSampler(valDataset)
    testSampler = RandomSampler(testDataset)

    trainDataloader = DataLoader(trainDataset, sampler = trainSampler, batch_size = 32, collate_fn = utils.collate_function)
    valDataloader = DataLoader(valDataset, sampler = valSampler, batch_size = 16, collate_fn = utils.collate_function)
    testDataloader = DataLoader(testDataset, sampler = testSampler, batch_size = 4, collate_fn = utils.collate_function)
    if opt.model_mode == 'single':
        model = RedisSingleDNN(opt.topk+5,2).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr = opt.lr, weight_decay = 0.01)
    elif opt.model_mode == 'twice':
        model = RedisTwiceDNN(opt.topk+5,2).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr = opt.lr, weight_decay = 0.01)
    elif opt.model_mode == "double":
        model, optimizer = dict(), dict()
        model['Totals_Ops_sec'] = RedisSingleDNN(opt.topk+5,1).to(DEVICE)
        model['Totals_p99_Latency'] = RedisSingleDNN(opt.topk+5,1).to(DEVICE)
        optimizer['Totals_Ops_sec'] = AdamW(model['Totals_Ops_sec'].parameters(), lr = opt.lr, weight_decay = 0.01)
        optimizer['Totals_p99_Latency'] = AdamW(model['Totals_p99_Latency'].parameters(), lr = opt.lr, weight_decay = 0.01)
    return model, optimizer, trainDataloader, valDataloader, testDataloader


def fitness_function(solution, args, model):
    solDataset = RedisDataset(solution,np.zeros((len(solution),2)))
    solDataloader = DataLoader(solDataset,shuffle=False,batch_size=args.n_pool,collate_fn=utils.collate_function)

    model.eval()

    fitness = np.array([])
    with torch.no_grad():
        for _, batch in enumerate(solDataloader):
            knobs_with_info = batch[0].to(DEVICE)
            fitness_batch = model(knobs_with_info).detach().cpu().numpy()
            if len(fitness) == 0:
                fitness = fitness_batch
            else:
                fitness = np.vstack([fitness,fitness_batch])
    return np.array(fitness)


def prepareForGA(args,top_k_knobs):
    with open("../data/workloads_info.json",'r') as f:
        workload_info = json.load(f)
    target_workload_info = np.array(workload_info[args.target])

    knobs_path = os.path.join(DATA_PATH, "configs")
    if args.persistence == "RDB":
        knob_data, _ = knobs.load_knobs(knobs_path)
    elif args.persistence == "AOF":
        _, knob_data = knobs.load_knobs(knobs_path)

    target_external_data, _ = knobs.load_metrics(metric_path = os.path.join(DATA_PATH,f'workload{args.target}',f'result_{args.persistence.lower()}_external_{args.target}.csv'),
                                    labels = knob_data['rowlabels'],
                                    metrics = ['Totals_Ops/sec', 'Totals_p99_Latency'])
    
    top_k_knobs = pd.DataFrame(knob_data['data'], columns = knob_data['columnlabels'])[top_k_knobs]                                 
    target_external_data = pd.DataFrame(target_external_data['data'], columns = ['Totals_Ops/sec', 'Totals_p99_Latency'])
    target_workload_infos = pd.DataFrame([target_workload_info],columns = workload_info['info'])

    top_k_knobs['tmp'] = 1
    target_workload_infos['tmp'] = 1

    knobWithworkload = pd.merge(top_k_knobs,target_workload_infos,on=['tmp'])
    knobWithworkload = knobWithworkload.drop('tmp',axis=1)

    scaler_X = MinMaxScaler().fit(knobWithworkload)
    scaler_y = StandardScaler().fit(target_external_data)

    return knobWithworkload, target_external_data, scaler_X, scaler_y
