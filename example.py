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


# Create datasets
mus = [0, -1, 1]
ndata = [1000, 1000, 1000]
data_list = [np.random.normal(mus[i], 1, ndata[i]) for i in range(len(mus))]

# Fill invcdfs
invcdfs = utils.invcdfs_from_data(data_list, nsamples_icdf=10000)

# Plot invcdf
utils.plot_invcdfs(invcdfs)

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

# Iterate gossip algorithm
network.iterate_async_gossip(100)


# Extract solitary models and learnt full models from network
models_sol, models_final = utils.extract_models_continuous(network,
                                                           start_cdf=-6,
                                                           stop_cdf=6,
                                                           nsamples_cdf=10000,
                                                           nbins=50)

# UNE FOIS QUE TU AS CA PAS BESOIN DE TOUCHER A AUTRE CHOSE

# Tu as accès aux cdf inverse :
# Plot learnt invcdf for the agent 0
plt.figure()
plt.plot(models_final[0].qs, models_final[0].invcdf)

# Aux cdf tout court
plt.figure()
plt.plot(models_final[0].ts, models_final[0].cdf)

# Aux pdfs: l'attribut rho est une fonction python de densité approximative
plt.figure()
plt.plot(models_final[0].ts, [models_final[0].rho(t) for t in models_final[0].ts])
