import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import importlib

import inv_cdfs as icdf
import agents_network as anet

importlib.reload(icdf)
importlib.reload(anet)


# Create empty InvCdf instances
invcdf1 = icdf.InvCdf(nsamples_icdf=10000)
invcdf2 = icdf.InvCdf(nsamples_icdf=10000)
invcdf3 = icdf.InvCdf(nsamples_icdf=10000)


# Fill them with beta distributions
# Parameters for the beta distributions
a1, b1 = 0.5, 0.5
a2, b2 = 0.1, 0.8
a3, b3 = 0.8, 0.1
# Instantiate the beta distributions
dist1 = stats.beta(a1, b1)
dist2 = stats.beta(a2, b2)
dist3 = stats.beta(a3, b3)


# Fill InvCdf instances with the beta distribution
invcdf1.fill_from_scipy(dist1)
invcdf2.fill_from_scipy(dist2)
invcdf3.fill_from_scipy(dist3)
# Plot the invcdfs
plt.figure()
plt.plot(invcdf1.qs, invcdf1.invcdf, label=0)
plt.plot(invcdf1.qs, invcdf2.invcdf, label=1)
plt.plot(invcdf1.qs, invcdf3.invcdf, label=2)
plt.legend()
plt.title("Initial inverse cdfs")


# Create agent network
# Number of agents in the network
npeers = 3
# Initialize neighbors
# Initialize agents with the beta distributions as model sol
agent1 = anet.Agent(npeers, 0, invcdf1)
agent2 = anet.Agent(npeers, 1, invcdf2)
agent3 = anet.Agent(npeers, 2, invcdf3)
# Stack agents in a list
agents = [agent1, agent2, agent3]

# Initialize weight matrix (here we take a complete graph)
W = np.ones((npeers, npeers)) - np.eye(npeers)
# Confidence of 1 in every model
C = np.ones(npeers)

# Build network
network = anet.AgentNetwork(W, C, agents, mu=1)

# Monitor iteration of asynchronous gossip algorithm
k1 = 3
k2 = 3
fig, axes = plt.subplots(nrows=k1, ncols=k2)
for t in range(0, k1 * k2):
    if t == 0:
        for i in range(0, npeers):
            axes[0, 0].plot(invcdf1.qs, network.agents[i].models_matrix[:, i], label=i)
        axes[0, 0].legend()
        axes[0, 0].set_title("t = 0")
    else:
        network.async_gossip_step()
        for i in range(0, npeers):
            axes[t // k1, t % k2].plot(invcdf1.qs, network.agents[i].models_matrix[:, i], label=i)
        axes[t // k1, t % k2].legend()
        axes[t // k1, t % k2].set_title("t = " + str(t))



# Run asynchronous gossip algorithm
network.iterate_async_gossip(100)
# Update finale model for each agents
network.update_invcdf_models()
# Plot propagated inverse cdfs
plt.figure()
for i in range(0, npeers):
    plt.plot(invcdf1.qs, network.agents[i].models_matrix[:, i], label=i)
plt.legend()
plt.title("Propagated inverse cdfs")
# Plot propagated cdfs
plt.figure()
for i in range(0, npeers):
    x, y = network.agents[i].invcdf_model.get_cdf(0, 1, 500)
    plt.plot(x, y, label=i)
plt.legend()
plt.title("Propagated cdfs")


