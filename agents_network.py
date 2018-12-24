import numpy as np


class Agent:

    def __init__(self, npeers, number, invcdf_sol):
        """
        Params:
            npeers (int): number of agents (including the agent itself) in the network
            number (int): the indice of the agent in the network
            invcdf_sol (inv_cdfs.InvCdf): the inverse cdf encoding the agents' local model (g_sol)
        """
        self.invcdf_sol = invcdf_sol
        self.number = number
        self.npeers = npeers
        self.models_matrix = np.zeros((invcdf_sol.qs.shape[0], npeers))
        self.models_matrix[:, number] = self.invcdf_sol.y

    def update_exterior_model(self, j, new_ext_model):
        """
        Params:
            j (int): number of the agent which exterior model we update
            new_ext_model (np.ndarray): new model, shape = (self.inv_cdf_sol.y.shape[0], )
        """
        self.models_matrix[:, j] = new_ext_model

    def get_model(self):
        return self.models_matrix[:, self.number]

    def get_model_sol(self):
        return self.invcdf_sol.y

    def set_model(self, new_model):
        self.models_matrix[:, self.number] = new_model

    def update_model(self, w, c, alpha):
        d = np.sum(w)
        new_model = alpha * np.sum((w / d) * self.models_matrix, axis=1)
        new_model += (1 - alpha) * c * self.invcdf_sol.y
        new_model *= 1 / (alpha + (1 - alpha) * c)
        self.set_model(new_model)


class AgentNetwork:

    def __init__(self, W, C, agents, mu):
        self.W = W
        self.C = C
        self.agents = agents
        self.mu = mu
        self.alpha = 1 / (1 + mu)

    def get_neighbors(self, i):
        return np.argwhere(self.W[i, :] != 0)[:, 0]

    def communication_step(self, i, j):
        self.agents[i].update_exterior_model(j, self.agents[j].get_model())
        self.agents[j].update_exterior_model(i, self.agents[i].get_model())

    def update_step(self, i, j):
        self.agents[i].update_model(self.W[i, :], self.C[i], self.alpha)
        self.agents[j].update_model(self.W[j, :], self.C[j], self.alpha)

    def draw_agent(self):
        return np.random.randint(0, self.W.shape[0])

    def draw_neighbor(self, i):
        return np.random.choice(self.get_neighbors(i), 1)[0]

    def async_gossip_step(self):
        i = self.draw_agent()
        j = self.draw_neighbor(i)
        self.communication_step(i, j)
        self.update_step(i, j)

    def iterate_async_gossip(self, nit):
        for t in range(0, nit):
            self.async_gossip_step()



