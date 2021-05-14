import numpy as np
from sklearn.gaussian_process.kernels import *
KERNEL_TYPE = ["RBF","Dot","ExpSine","White","RBF with C"]
LENGTH_SCALE = np.arange(0.5,1.5,0.1)
NOISE_LEVEL = np.arange(0.1,1.5,0.1)
PERIODICITY = np.arange(0.5,1.5,0.1)
SIGMA = np.arange(0.5,1.5,0.1)
CONSTANT_VALUE = np.arange(1,10,1)

options = []
for KERNEL in KERNEL_TYPE:
    componenet = {}
    if KERNEL == "RBF":
        componenet[KERNEL] = [{"LENGTH_SCALE": LENGTH_SCALE}]
    elif KERNEL == "Dot":
        componenet[KERNEL] = [{"SIGMA": SIGMA}]
    elif KERNEL == "ExpSine":
        componenet[KERNEL] = [{"LENGTH_SCALE":LENGTH_SCALE,"PERIODICITY":PERIODICITY}]
    elif KERNEL == "White":
        componenet[KERNEL] = [{"NOISE_LEVEL":NOISE_LEVEL}]
    elif KERNEL == "RBF with C":
        componenet[KERNEL] = [{"LENGTH_SCALE":LENGTH_SCALE,"CONSTANT_VALUE":CONSTANT_VALUE}]
    else:
        raise Exception("Not Define Kernel")
    options.append(componenet)

print(options)
