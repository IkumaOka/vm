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

    def __call__(self, z, log_q):
        z += self.u * self.h((z*self.w).sum(-1).unsqueeze(-1) + self.b)
        psi = self.h_prime((z*self.w).sum(-1).unsqueeze(-1) + self.b) * self.w
        det_jacob = torch.abs(1 + (psi*self.u).sum(-1))
        log_q -= torch.log(1e-7 + det_jacob)
        return z, log_q