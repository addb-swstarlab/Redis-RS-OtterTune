import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target', type = int, default = 1, help='Target Workload')
parser.add_argument('--persistence', type = str, choices = ["RDB","AOF"], default = 'RDB', help='Choose Persistant Methods')
parser.add_argument('--path',type= str)
parser.add_argument('--num', type = str)

args = parser.parse_args()
