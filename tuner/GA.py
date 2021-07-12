import os
import sys
import copy
import logging
import argparse
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd

import utils

sys.path.append('../')

from models.steps import (fitness_function,prepareForGA)
from models.dnn import RedisSingleDNN, RedisTwiceDNN

parser = argparse.ArgumentParser()
parser.add_argument('--target', type = str, default = '1', help='Target Workload')
parser.add_argument('--persistence', type = str, choices = ["RDB","AOF"], default = 'RDB', help='Choose Persistant Methods')
parser.add_argument('--topk', type = int, default=4,)
parser.add_argument('--path',type= str)
parser.add_argument('--num', type = str)
parser.add_argument('--n_pool',type = int, default = 32)
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
        model = RedisSingleDNN(args.topk+5,2)
        model.load_state_dict(torch.load(os.path.join('./model_save',args.path,'model_{}.pt'.format(args.num))))
    pruned_configs, external_data, scaler_X, scaler_y = prepareForGA(args,top_k_knobs)
    temp_configs = pd.concat([pruned_configs,external_data],axis=1)
    temp_configs = temp_configs.sort_values(["Totals_Ops/sec","Totals_p99_Latency"], ascending=[False,True])
    target = temp_configs[["Totals_Ops/sec","Totals_p99_Latency"]].values[0]
    configs = temp_configs.drop(columns=["Totals_Ops/sec","Totals_p99_Latency"])
    n_configs = top_k_knobs.shape[0]
    n_pool_half = args.n_pool//2
    mutation = int(n_configs)

    current_solution_pool = configs[:args.n_pool].values
    target = scaler_y.transform([target]*args.n_pool)
    for i in tqdm(range(args.n_generation)):
        scaled_pool = scaler_X.transform(current_solution_pool)
        predicts = fitness_function(scaled_pool, args, model)
        #fitness = scaler_y.inverse_transform(predicts)

        def mse_loss(target,predict):
            loss = (target-predict)**2
            return loss[0][0]-loss[0][1]
        idx_fitness = []
        for value in predicts:
            idx_fitness.append(mse_loss(target,value))
        idx_fitness = np.argsort(idx_fitness)[:n_pool_half]
        idx_fitness = idx_fitness[:n_pool_half]
        best_solution_pool = current_solution_pool[idx_fitness,:]
        fitness = scaler_y.inverse_transform(predicts)
        if i % 1000 == 999:
            print(f"[{i+1:3d}/{args.n_generation:3d}] best fitness: {list(fitness[idx_fitness[0]])}")
        pivot = np.random.choice(np.arange(1,n_configs))
        new_solution_pool = np.zeros_like(best_solution_pool)
        for j in range(n_pool_half):
            new_solution_pool[j][:pivot] = best_solution_pool[j][:pivot]
            new_solution_pool[j][pivot:n_configs] = best_solution_pool[n_pool_half-1-j][pivot:n_configs]
            new_solution_pool[j][n_configs:] = current_solution_pool[0][n_configs:]
            import utils, random
            random_knobs = utils.make_random_option(top_k_knobs)
            knobs = list(random_knobs.values())
            random_knob_index = list(range(n_configs))
            random.shuffle(random_knob_index)
            random_knob_index = random_knob_index[:mutation]
            for k in range(len(random_knob_index)):
                new_solution_pool[j][random_knob_index[k]] = knobs[random_knob_index[k]]
        current_solution_pool = np.vstack([best_solution_pool, new_solution_pool])
    final_solution_pool = pd.DataFrame(best_solution_pool)
    print(final_solution_pool)

if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()
