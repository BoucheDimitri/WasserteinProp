    def update_exterior_model(self, j, new_ext_model):
        """
        Update one columns of the agents models matrix with a new model
        Params:
            j (int): number of the agent which exterior model we update
            new_ext_model (np.ndarray): new model, shape = (self.inv_cdf_sol.y.shape[0], )
        """
        self.models_matrix[:, j] = new_ext_model

    def get_model(self):
        """
        Getter for agent's model
        Returns:
            np.ndarray: The agent's model
        """
        return self.models_matrix[:, self.number]

    def get_model_sol(self):
        """
        Getter for agent's original model (model "sol")
        Returns:
            np.ndarray: The agent's original model
        """
        return self.invcdf_sol.y

    def set_model(self, new_model):
        """
        Setter for the agents model in its models_matrix
        """
        self.models_matrix[:, self.number] = new_model

    def update_model(self, w, c, alpha):
        """
        Update agent's model in its models_matrix using the neighborhood model update formula
        Params:
            w (np.ndarray): row of the W matrix corresponding to the agent's number
            c (float): confidence in the agent's model
            alpha (float): alpha parameter for the update
        """
        d = np.sum(w)
        new_model = alpha * np.sum((w / d) * self.models_matrix, axis=1)
        new_model += (1 - alpha) * c * self.invcdf_sol.y
        
        
class AgentNetwork:

    def __init__(self, W, C, agents, mu):
        """
        Params:
            W (np.ndarray): Weights matrix for the graph, shape = (nagents, nagents)
            C (np.ndarray): Confidence in the models, shape = (nagents, )
            agents (list): List of agents_network.Agent of size nagents
            mu (float): Mu parameter
        """
        self.W = W
        self.C = C
        self.agents = agents
        self.mu = mu
        self.alpha = 1 / (1 + mu)

    def get_neighbors(self, i):
        """
        Params:
            i (int): Agent which neighbors we want
        Returns:
            np.ndarray: neighbors of i stacked in a np.ndarray
        """
        return np.argwhere(self.W[i, :] != 0)[:, 0]

    def communication_step(self, i, j):
        """
        Communication step from the article
        Params:
            i (int): first agent involved
            j (int): second agent involved
        """
        self.agents[i].update_exterior_model(j, self.agents[j].get_model())
        self.agents[j].update_exterior_model(i, self.agents[i].get_model())

    def update_step(self, i, j):
        """
        Update step from the article
        Params:
            i (int): first agent involved
            j (int): second agent involved
        """
        self.agents[i].update_model(self.W[i, :], self.C[i], self.alpha)
        self.agents[j].update_model(self.W[j, :], self.C[j], self.alpha)

    def draw_agent(self):
        """
        Draw agent uniformly at random
        Returns:
            int: index of an agent
        """
        return np.random.randint(0, self.W.shape[0])

    def draw_neighbor(self, i):
        """
        Draw a neighbor of an agent uniformly at random
        Params:
            i (int): Index of the agent from which neighbors we want to draw
        Returns:
            int: a neighbor of i drawn randomly
        """
        return np.random.choice(self.get_neighbors(i), 1)[0]

    def async_gossip_step(self):
        """
        A step of the asynchronous gossip algorithm
        """
        i = self.draw_agent()
        j = self.draw_neighbor(i)
        self.communication_step(i, j)
        self.update_step(i, j)

    def iterate_async_gossip(self, nit):
        """
        Iterate the asynchronous gossip algorithm
        """
        for t in range(0, nit):
            self.async_gossip_step()
