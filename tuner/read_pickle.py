import os
import pickle

with open("/home/jinhuijun/CDBTune/tuner/test_knob/eval_ddpg_1616552055.pkl","rb") as f:
	data = pickle.load(f)

print(data[0])
