import torch
import numpy as np
class NeRFToGaussianBridge:
    '''
    Implements the hybrid transition by sampling NeRF density field to initialize Gaussian primitives
    '''