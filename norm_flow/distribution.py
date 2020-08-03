import numpy as np
import matplotlib.pyplot as plt


class Distribution:
    def calc_prob(self, z):
        p = p.zeros(z.shape[0])
        return p

    def plot(self, size=5):
        side = np.linspace(-size, size, 1000)
        z1, z2 = np.meshgrid(side, side)
        shape = z1.shape
        z1 = z1.ravel()
        z2 = z2.ravel()
        z = np.c_[z1, z2]
        probability = self.calc_prob(z).reshape(shape)
        plt.figure(figsize=(6, 6))
        plt.imshow(probability)
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        plt.show()


class NormalDistribution2D(Distribution):
    def sample(self, sample_num, n_dim):
        z = np.random.randn(sample_num, n_dim)
