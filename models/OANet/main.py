import os
import argparse
import pandas as pd
import numpy as np
from models.OANet.models.train import train_Net
from models.OANet.models.utils import get_logger
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.system('clear')
## pareser 에서 argparse대신에 다른 input 받아야 함. 수정요망!
##python main.py --train --mode {reshape or single } --external {external matrix} --dot --lamb {lamb} --hidden_size {hidden size} --group_size {group size} --epochs {epochs} --lr {learning rate}
parser = --train --mode reshape --external TIME
parser = argparse.ArgumentParser()
parser.add_argument('--external', type=str, choices=['TIME', 'RATE', 'WAF', 'SA'], help='choose which external matrix be used as performance indicator')
# parser.add_argument('--kf', type=float, default=3, help='Define split number for K-Folds cross validation')
parser.add_argument('--mode', type=str, default='reshape', help='choose which model be used on fitness function')
parser.add_argument('--hidden_size', type=int, default=16, help='Define model hidden size')
parser.add_argument('--group_size', type=int, default=32, help='Define model gruop size')
parser.add_argument('--dot', action='store_true', help='if trigger, model use loss term, dot')
parser.add_argument('--lamb', type=float, default=0.1, help='define lambda of loss function' )
parser.add_argument('--lr', type=float, default=0.01, help='Define learning rate')  
parser.add_argument('--act_function', type=str, default='Sigmoid', help='choose which model be used on fitness function')   
parser.add_argument('--epochs', type=int, default=30, help='Define train epochs')   
parser.add_argument('--batch_size', type=int, default=64, help='Define model batch size')
parser.add_argument('--train', action='store_true', help='if trigger, model goes triain mode')
parser.add_argument('--eval', action='store_true', help='if trigger, model goes eval mode')




opt = parser.parse_args()

if not os.path.exists('logs'):
    os.mkdir('logs')
    
logger, log_dir = get_logger(os.path.join('./logs'))

## print parser info
logger.info("## model hyperparameter information ##")
for i in vars(opt):
    logger.info(f'{i}: {vars(opt)[i]}')

# KNOB_PATH = 'data/knobs.csv'
# EXTERNAL_PATH = 'data/external'
# WK_NUM = 4


##변경함
DATA_PATH = "/home/capstone2201/data" #수정
WK_NUM = 3
# MODE = 'reshape'
# batch_size = 64
# lr = 0.01
# epochs = 30

# group_dim = 8



def main():
    logger.info("## get raw datas ##")

    data = []
    one_hot = np.eye(WK_NUM)

    for wk in range(WK_NUM):
        ##받아와야하는 부분
        data_wk = pd.read_csv(os.path.join(DATA_PATH, f'rocksdb_benchmark_{str(wk)}.csv'))
        oh = np.repeat([one_hot[wk]], len(data_wk), axis=0)
        oh = pd.DataFrame(oh)
        data_wk = pd.concat((data_wk, oh), axis=1)
        data.append(data_wk)

    data = pd.concat(data)
    data = data.reset_index(drop=True)

    logger.info('## get raw datas DONE ##')


    if opt.train:
        r2, pcc, ci, MSE, true, pred, df_pred = train_Net(logger, data=data, METRIC=opt.external, MODE=opt.mode, 
        batch_size=opt.batch_size, lr=opt.lr, epochs=opt.epochs, 
        hidden_dim=opt.hidden_size, group_dim=opt.group_size, WK_NUM=WK_NUM, dot=opt.dot, lamb=opt.lamb) 

        logger.info(f'\npred = \n{pred[:5]}, {np.exp(pred[:5])}')
        logger.info(f'\ntrue = \n{true[:5]}, {np.exp(true[:5])}')
        logger.info(f'\naccuracy(mean_squared_error) = {mean_squared_error(true, pred)}\naccuracy(mean_absolute_error) = {mean_absolute_error(true, pred)}')
        logger.info(f'\nMetric : {opt.external}')
        logger.info(f'  (r2 score) = {r2:.4f}')
        logger.info(f'  (pcc score) = {pcc:.4f}')
        logger.info(f'  (ci score) = {ci:.4f}')
        logger.info(f'  (MSE score) = {MSE:.4f}')

            
    elif opt.eval:
        logger.info('## EVAL MODE ##')


if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()
