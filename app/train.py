import sys
import os
import time

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

def load_checkpoint(model, path):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['state_dict'])
    epoch = state_dict['epoch']+1
    return model,epoch