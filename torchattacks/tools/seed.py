import numpy as np
import random
import os
import torch
# import tensorflow as tf

def random_seed(seed=10001):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # tf.random.set_seed(seed)
