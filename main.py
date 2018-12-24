import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import importlib

import inv_cdfs as icdf
import agents_network as anet

importlib.reload(icdf)
importlib.reload(anet)


# Create empty InvCdf instances
invcdf1 = icdf.InvCdf()
invcdf2 = icdf.InvCdf()
invcdf3 = icdf.InvCdf()
invcdf4 = icdf.InvCdf()


# Fill them with beta distributions
# Parameters for the beta distributions
a1, b1 = 0.5, 0.5
a2, b2 = 0.1, 0.8
a3, b3 = 0.8, 0.1
a4, b4 = 0.4, 0.6
# Instantiate the beta distributions
beta1 = stats.beta(a1, b1)
beta2 = stats.beta(a2, b2)
beta3 = stats.beta(a3, b3)
beta4 = stats.beta(a4, b4)
# Fill InvCdf instances with the beta distribution
invcdf1.fill_from_scipy(beta1)
invcdf2.fill_from_scipy(beta2)
invcdf3.fill_from_scipy(beta3)
invcdf4.fill_from_scipy(beta4)
# Plot the invcdfs
plt.figure()
plt.plot(invcdf1.y, label=0)
plt.plot(invcdf2.y, label=1)
plt.plot(invcdf3.y, label=2)
plt.plot(invcdf4.y, label=3)
plt.legend()
plt.title("Initial inverse cdfs")


# Create agent network
# Number of agents in the network
npeers = 4
# Initialize neighbors
# Initialize agents with the beta distributions as model sol
agent1 = anet.Agent(npeers, 0, invcdf1)
agent2 = anet.Agent(npeers, 1, invcdf2)
agent3 = anet.Agent(npeers, 2, invcdf3)
agent4 = anet.Agent(npeers, 3, invcdf4)
# Stack agents in a list
agents = [agent1, agent2, agent3, agent4]
# Initialize weight matrix (here we take a complete graph)
W = np.ones((npeers, npeers)) - np.eye(npeers)
# Confidence of 1 in every model
C = np.ones(npeers)
# Build network
network = anet.AgentNetwork(W, C, agents, mu=1)
# Run asynchronous gossip algorithm
network.iterate_async_gossip(100)
# Plot propagated cdfs
plt.figure()
for i in range(0, npeers):
    plt.plot(network.agents[i].models_matrix[:, i], label=i)
plt.legend()
plt.title("Propagated inverse cdfs")