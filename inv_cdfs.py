import numpy as np


class InvCdf:

    def __init__(self, nsamples=500):
        """
        Params:
            nsamples (int): Number of discretization samples
        """
        self.nsamples = nsamples
        self.qs = np.linspace(0, 1, self.nsamples)
        self.y = np.zeros(nsamples)

    def fill_from_scipy(self, distrib):
        """
        Fill self.y with discretized quantile function from a scipy distribution

        Params;
            distrib (scipy.stats._distn_infrastructure.rv_frozen): The scipy distribution
        """
        for i in range(0, self.nsamples):
            self.y[i] = distrib.ppf(self.qs[i])

    def fill_from_data(self, data):
        """
        Fill self.y with discretized empirical quantile function from a sample

        Params:
            data (np.ndarray): the datasample to infer the empirical quantile function from
        """
        for i in range(0, self.nsamples):
            self.y[i] = np.percentile(data, 100 * self.qs[i])


