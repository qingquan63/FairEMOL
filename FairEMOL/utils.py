import numpy as np
import random
from scipy.optimize import minimize # for loss func minimization
from copy import deepcopy
import sys

SEED = 1122334455
random.seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def add_intercept(x):
    """ Add intercept to the data before linear classification """
    m,n = x.shape
    intercept = np.ones(m).reshape(m, 1) # the constant b
    return np.concatenate((intercept, x), axis = 1)
