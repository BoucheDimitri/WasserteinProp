import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import importlib

import inv_cdfs as icdf
import agents_network as anet

importlib.reload(icdf)
importlib.reload(anet)


def invcdfs_from_data(data_list, nsamples_icdf):
    npeers = len(data_list)
    invcdfs = [icdf.InvCdf(nsamples_icdf) for i in range(npeers)]
    for i in range(npeers):
        invcdfs[i].fill_from_data(data_list[i])
    return invcdfs


def invcdfs_from_distrib(distrib_list, nsamples_icdf):
    npeers = len(distrib_list)
    invcdfs = [icdf.InvCdf(nsamples_icdf) for i in range(npeers)]
    for i in range(npeers):
        invcdfs[i].fill_from_scipy(distrib_list[i])
    return invcdfs


def plot_invcdfs(invcdfs):
    n = len(invcdfs)
    plt.figure()
    for i in range(n):
        plt.plot(invcdfs[i].qs, invcdfs[i].invcdf, label=i)
    plt.legend()


def build_anet(W, C, invcdfs, mu=1):
    npeers = W.shape[0]
    agents = [anet.Agent(npeers, k, invcdfs[k]) for k in range(npeers)]
    network = anet.AgentNetwork(W, C, agents, mu)
    return network


def plot_aga_iterations(network, k1, k2):
    npeers = network.agents[0].npeers
    fig, axes = plt.subplots(nrows=k1, ncols=k2, sharex=True, sharey=True)
    qs = network.agents[0].invcdf_sol.qs
    for t in range(0, k1 * k2):
        if t == 0:
            for i in range(0, npeers):
                axes[0, 0].plot(qs, network.agents[i].models_matrix[:, i], label=i)
            axes[0, 0].set_title("t = 0")
        else:
            network.async_gossip_step()
            for i in range(0, npeers):
                axes[t // k2, t % k2].plot(qs, network.agents[i].models_matrix[:, i], label=i)
            axes[t // k2, t % k2].set_title("t = " + str(t))
    plt.suptitle("Inverse CDFs during AGA iterations")


def extract_models_discrete(network, start_cdf, stop_cdf, nsamples_cdf, thresh=0.0002):
    # Update finale model for each agents
    network.update_invcdf_models()
    network.fill_cdfs(start_cdf, stop_cdf, nsamples_cdf)
    network.fill_rho_discrete(thresh)
    npeers = network.W.shape[0]
    models_sol = [network.agents[i].invcdf_sol for i in range(npeers)]
    models_final = [network.agents[i].invcdf_model for i in range(npeers)]
    return models_sol, models_final


def extract_models_continuous(network, start_cdf, stop_cdf, nsamples_cdf, nbins=30):
    # Update finale model for each agents
    network.update_invcdf_models()
    network.fill_cdfs(start_cdf, stop_cdf, nsamples_cdf)
    network.fill_rho_continuous(nbins)
    npeers = network.W.shape[0]
    models_sol = [network.agents[i].invcdf_sol for i in range(npeers)]
    models_final = [network.agents[i].invcdf_model for i in range(npeers)]
    return models_sol, models_final


def plot_invcdf_cdf_pdf_continuous(models_sol, models_final):
    fig, axes = plt.subplots(nrows=2, ncols=3)
    qs = models_sol[0].qs
    ts = models_sol[0].ts
    for i in range(len(models_sol)):
        axes[0, 0].plot(qs, models_sol[i].invcdf, label=i)
        axes[1, 0].plot(qs, models_final[i].invcdf, label=i)
    axes[0, 0].legend()
    axes[0, 0].set_title("Solitary Inverse CDF")
    axes[1, 0].legend()
    axes[1, 0].set_title("Learnt Inverse CDF")
    for i in range(len(models_sol)):
        axes[0, 1].plot(ts, models_sol[i].cdf, label=i)
        axes[1, 1].plot(ts, models_final[i].cdf, label=i)
    axes[0, 1].legend()
    axes[0, 1].set_title("Solitary CDF")
    axes[1, 1].legend()
    axes[1, 1].set_title("Learnt CDF")
    for i in range(len(models_sol)):
        axes[0, 2].plot(ts, [models_sol[i].rho(t) for t in ts], label=i)
        axes[1, 2].plot(ts, [models_final[i].rho(t) for t in ts], label=i)
    axes[0, 2].legend()
    axes[0, 2].set_title("Solitary PDF")
    axes[1, 2].legend()
    axes[1, 2].set_title("Learnt PDF")


def plot_invcdf_cdf_pdf_discrete(models_sol, models_final):
    # fig, axes = plt.subplots(nrows=2, ncols=3)
    fig, axes = plt.subplots(nrows=2, ncols=2)
    qs = models_sol[0].qs
    ts = models_sol[0].ts
    for i in range(len(models_sol)):
        axes[0, 0].plot(qs, models_sol[i].invcdf, label=i)
        axes[1, 0].plot(qs, models_final[i].invcdf, label=i)
    axes[0, 0].legend()
    axes[0, 0].set_title("Solitary Inverse CDF")
    axes[1, 0].legend()
    axes[1, 0].set_title("Learnt Inverse CDF")
    for i in range(len(models_sol)):
        axes[0, 1].plot(ts, models_sol[i].cdf, label=i)
        axes[1, 1].plot(ts, models_final[i].cdf, label=i)
    axes[0, 1].legend()
    axes[0, 1].set_title("Solitary CDF")
    axes[1, 1].legend()
    axes[1, 1].set_title("Learnt CDF")
    # for i in range(len(models_sol)):
    #     axes[0, 2].plot(models_sol[i].rho[0], models_sol[i].rho[1], label=i)
    #     axes[1, 2].scatter(models_final[i].rho[0], models_final[i].rho[1], label=i)
    # axes[0, 2].legend()
    # axes[0, 2].set_title("Solitary PDF")
    # axes[1, 2].legend()
    # axes[1, 2].set_title("Learnt PDF")


def plot_pdf_discrete(models_sol, models_final):
    fig, axes = plt.subplots(nrows=2, ncols=3)
    for i in range(len(models_final)):
        axes[1, i].bar(models_final[i].rho[0], models_final[i].rho[1], width=0.1, color="C" + str(i))
        axes[1, i].set_title("Learned PDF")
        axes[0, i].bar(models_sol[i].rho[0], models_sol[i].rho[1], width=0.1, color="C" + str(i))
        axes[0, i].set_title("Solitary PDF")
    plt.suptitle("PDF comparison - Solitary and learnt")


