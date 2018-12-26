import numpy as np
import bisect


class InvCdf:

    def __init__(self, nsamples_icdf=10000, invcdf_vec=None, density_type="continuous"):
        """
        Params:
            nsamples (int): Number of discretization samples
            invcdf_vec (np.ndarray): if given, self.y is filled directly with invcdf_vec
        """
        if isinstance(invcdf_vec, np.ndarray):
            self.nsamples_icdf = invcdf_vec.shape[0]
            self.invcdf = invcdf_vec.copy()
            self.qs = np.linspace(0, 1, self.nsamples_icdf)
        else:
            self.nsamples_icdf = nsamples_icdf
            self.qs = np.linspace(0, 1, self.nsamples_icdf)
            self.invcdf = np.zeros(nsamples_icdf)
        self.density_type = density_type
        self.nsamples_cdf = None
        self.cdf = None
        self.ts = None
        self.rho = None

    def fill_from_scipy(self, distrib):
        """
        Fill self.y with discretized quantile function from a scipy distribution

        Params;
            distrib (scipy.stats._distn_infrastructure.rv_frozen): The scipy distribution
        """
        for i in range(0, self.nsamples_icdf):
            self.invcdf[i] = distrib.ppf(self.qs[i])

    def fill_from_data(self, data):
        """
        Fill self.y with discretized empirical quantile function from a sample

        Params:
            data (np.ndarray): the datasample to infer the empirical quantile function from
        """
        for i in range(0, self.nsamples_icdf):
            self.invcdf[i] = np.percentile(data, 100 * self.qs[i])

    def invert(self, t):
        """
        Discretized inverse of inverse-cdf : discretized cdf

        Params:
            t (float): where to evaluate the cdf

        Returns:
            float: approximation of cdf in t
        """
        loc = bisect.bisect(self.invcdf, t)
        if loc >= self.qs.shape[0]:
            return 1
        elif loc == 0:
            return 0
        else:
            return self.qs[loc]

    def fill_cdf(self, start, stop, nsamples_cdf=10000):
        """
        Apply self.invert on a linspace

        Params:
            start (float): start of linspace
            stop (float): end of linspace
            nsamples (float): number of points in linspace

        Returns:
            tuple: linspace, cdf on linspace
        """
        self.nsamples_cdf = nsamples_cdf
        self.ts = np.linspace(start, stop, nsamples_cdf)
        self.cdf = np.array([self.invert(t) for t in self.ts])

    def get_pdf(self):
        return self.ts[1:], np.diff(self.cdf)

    def fill_rho_discrete(self):
        ts, pdf = self.get_pdf()
        inds = np.argwhere(pdf != 0)[:, 0]
        vals = ts[inds]
        probs = pdf[inds]
        self.rho = vals, probs

    def fill_rho_continuous(self):
        ts, pdf = self.get_pdf()

        def rho_continuous(x):
            y = np.interp(np.array([x]), ts, pdf)
            return y[0]

        self.rho = rho_continuous

    def fill_rho(self):
        if self.density_type == "continuous":
            self.fill_rho_continuous()
        else:
            self.fill_rho_discrete()




