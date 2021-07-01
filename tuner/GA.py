import os
import sys
import copy
import logging
import argparse

import torch
import numpy as np
import pandas as pd

import utils

sys.path.append('../')

from models.steps import (fitness_function,prepareForGA)
from models.dnn import RedisSingleDNN, RedisTwiceDNN

parser = argparse.ArgumentParser()
parser.add_argument('--target', type = int, default = 1, help='Target Workload')
parser.add_argument('--persistence', type = str, choices = ["RDB","AOF"], default = 'RDB', help='Choose Persistant Methods')
parser.add_argument('--topk', type = int, default=4,)
parser.add_argument('--path',type= str)
parser.add_argument('--num', type = str)
parser.add_argument('--n_pool',type = int, default = 512)
parser.add_argument('--n_generation', type=int, default=10000,)
parser.add_argument("--model_mode", type = str, default = 'single', help = "model mode")

args = parser.parse_args()

if not os.path.exists('save_knobs'):
    assert "Do this file after running main.py"

print("======================MAKE GA LOGGER====================")
logger, log_dir = utils.get_logger(os.path.join('./GA_logs'))

def main():
    top_k_knobs = np.load(os.path.join('./save_knobs',args.path,f"knobs_{args.topk}.npy"))
    if args.model_mode == 'single':
        model = RedisSingleDNN(9,2)
        model.load_state_dict(torch.load(os.path.join(args.path,'model_{}.pt'.format(args.num))))
    pruned_configs, external_data, scaler_X, scaler_y = prepareForGA(args,top_k_knobs)
    temp_configs = pd.concat([pruned_configs,external_data],axis=1)
    configs = temp_configs.sort_values(["Totals_Ops/sec","Totals_p99_Latency"], ascending=[False,True]).drop(columns=["Totals_Ops/sec","Totals_p99_Latency"])
    n_configs = configs.shape[1]
    n_pool_half = args.n_pool//2
    mutation = int(n_configs * 0.4)

    current_solution_pool = configs[:args.n_pool].values

    for i in range(args.n_generation):
        scaled_pool = scaler_X.transform(current_solution_pool)
        predicts = fitness_function(scaled_pool, model)
        fitness = scaler_y.inverse_transform(predicts)

    



if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()
