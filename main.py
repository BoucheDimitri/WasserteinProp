import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import importlib

import inv_cdfs as icdf
import agents_network as anet
import utils

importlib.reload(icdf)
importlib.reload(anet)
importlib.reload(utils)

# Plot parameters
plt.rcParams.update({"font.size": 22})
plt.rcParams.update({"lines.linewidth": 3})
plt.rcParams.update({"lines.markersize": 10})


# ############## CREATE INV CDF FOR BETA DISTRIBUTION ##################################################################
# Create beta distributions as an example
a, b = [0.5, 0.1, 0.8], [0.5, 0.8, 0.1]
betas = [stats.beta(a[i], b[i]) for i in range(len(a))]

# Fill InvCdf instances with the beta distributions
invcdfs = utils.invcdfs_from_distrib(betas, nsamples_icdf=10000)

# Plot the invcdfs
utils.plot_invcdfs(invcdfs)


# ######## EXAMPLE OF CONVERGENCE OF THE ALGORITHM #####################################################################
# Number of agents in the network
npeers = 3

# Initialize weight matrix (here we take a complete graph)
W = np.ones((npeers, npeers)) - np.eye(npeers)
# Confidence of 1 in every model
C = np.ones(npeers)

# mu parameter
mu = 1

# Build network
network = utils.build_anet(W, C, invcdfs, mu)

# Monitor iteration of asynchronous gossip algorithm
k1 = 3
k2 = 4
utils.plot_aga_iterations(network, k1, k2)


# ######## PROPAGATED CDF AND PDF: BETA CASE ###########################################################################
# Number of agents in the network
npeers = 3

# Initialize weight matrix (here we take a complete graph)
W = np.ones((npeers, npeers)) - np.eye(npeers)
# Confidence of 1 in every model
C = np.ones(npeers)

# mu parameter
mu = 1

# Build network
network = utils.build_anet(W, C, invcdfs, mu)

# Run asynchronous gossip algorithm
network.iterate_async_gossip(100)

# Extract final full models from network
models_sol, models_final = utils.extract_models_continuous(network,
                                                           start_cdf=0,
                                                           stop_cdf=1,
                                                           nsamples_cdf=10000,
                                                           nbins=100)

utils.plot_invcdf_cdf_pdf_continuous(models_sol, models_final)


# ######### EXAMPLE WITH DISCRETE DISTRIBUTION : BERNOUILLI ############################################################

# Create binomial distributions as an example
ps = [0.3, 0.5, 0.8]
n = 5
bernouillis = [stats.bernoulli(ps[i]) for i in range(len(ps))]

# Fill InvCdf instances with the bernouilli distributions
invcdfs = utils.invcdfs_from_distrib(bernouillis, nsamples_icdf=10000)

# Plot the invcdfs
utils.plot_invcdfs(invcdfs)


# ########### PLOT ITERATIONS ##########################################################################################
# Number of agents in the network
npeers = 3

# Initialize weight matrix (here we take a complete graph)
W = np.ones((npeers, npeers)) - np.eye(npeers)
# Confidence of 1 in every model
C = np.ones(npeers)

# mu parameter
mu = 1

# Build network
network = utils.build_anet(W, C, invcdfs, mu)

# Monitor iteration of asynchronous gossip algorithm
k1 = 3
k2 = 4
utils.plot_aga_iterations(network, k1, k2)


# Number of agents in the network
npeers = 3


# ###################### PLOT INV CDF, CDF, PDF #######################################################################
# Build network
network = utils.build_anet(W, C, invcdfs, mu)

# Run asynchronous gossip algorithm
network.iterate_async_gossip(100)

# Extract final full models from network
models_sol, models_final = utils.extract_models_discrete(network,
                                                         start_cdf=-1,
                                                         stop_cdf=2,
                                                         nsamples_cdf=10000,
                                                         thresh=0.0002)

utils.plot_invcdf_cdf_pdf_discrete(models_sol, models_final)

utils.plot_pdf_discrete(models_sol, models_final)