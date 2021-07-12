from collections import OrderedDict
import itertools
from grid_main import grid_main
import utils
import os
import pandas as pd

def get_runs(hyperparams_dict):
    runs = []
    for v in itertools.product(*hyperparams_dict.values()):
        run = OrderedDict()
        for k, vv in zip(hyperparams_dict.keys(),v):
            run[k] =vv
        runs.append(run)
    return runs


if __name__ == '__main__':
    hyperparams_dict = OrderedDict(
        target = list(range(1,19)),
        persistence = ['RDB'],
        db = ['redis'],
        cluster = ['k-means','ms','gmm'],
        rki = ['lasso','RF','XGB'],
        topk = [4,8,12,16,20],
        model_mode = ['single','twice'],
        n_epochs = [150],
        lr = [1e-6],
    )
    runs = get_runs(hyperparams_dict)
    results_list = []
    print("Total runs: ",len(runs))
    for i, run in enumerate(runs):
        logger,log_dir = utils.get_logger(os.path.join('./logs'))
        results = grid_main(run, logger, log_dir)
        results_list.append(list(results.__dict__.values()))
    result_table = pd.DataFrame(results_list,columns=list(results.__dict__.keys()))
    result_table.to_csv("results_grid.csv")

        
