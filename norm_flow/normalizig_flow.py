import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf


class PlanarFlow:
    def __init__(self, dim):
        self.dim = dim
        self.h = lambda x: nn.Tanh(x)
        self.h_prime = lambda x: 1 - nn.Tanh(x)**2
        self.w = torch.tensor(tf.radom.truncated_normal(shape=(1, self.dim)))
        self.b = torch.zeros(1)
        self.u = torch.tensor(tf.radom.truncated_normal(shape=(1, self.dim)))
