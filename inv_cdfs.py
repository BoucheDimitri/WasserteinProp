import numpy as np


class InvCdf:

    def __init__(self, nsamples=500):
        self.nsamples = nsamples
        self.qs = np.linspace(0, 1, self.nsamples)
        self.y = np.zeros(nsamples)

    def fill_from_scipy(self, distrib):
        for i in range(0, self.nsamples):
            self.y[i] = distrib.ppf(self.qs[i])

    def fill_from_data(self, data):
        for i in range(0, self.nsamples):
            self.y[i] = np.percentile(data, 100 * self.qs[i])


