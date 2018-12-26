import numpy as np
import bisect
form scipy.stats import norm


class Obtenir_la_densite_de_probabilitees:
    
    def __init__(self, pts_discretisation, inv_cdf):
        """
        Params:
            pts_discretisation : Set of discretization points
            inv_cdf : fonction inv_cdf(agent,s)=g_s(v) où s est le point en lequel on calcule
        """
        self.pts_discretisation = pts_discretisation
        self.inv_cdf = inv_cdf
    
    ''' crée un échantillon de nsamples variables tirées par un agent'''
    def draw(self, agent, nsamples=1000):
        U = np.random.rand(nsamples)
        ''' ici self.pts_discretisation = les pts s_i '''
        func = np.vectorize(lambda u : bisect.bisect(self.pts_discretisation, u))
        S = func(U) # tous les points s_i pour lesquels on renvoie (g_{s_i}(v))
        return np.array([inv_cdf(agent,s) for s in S])
        
    ''' plutôt dans le cas discret '''
    def methode_MC(self, agent, nsamples=1000):
        echantillon = draw(agent, nsamples)
        ensemble_des_valeurs, Nb_occurences = np.unique(echantillon, return_counts=True)
        prob =  Nb_occurences/np.sum(Nb_occurences)
        return ensemble_des_valeurs, prob
    
    ''' pour la méthode à noyaux, on retourne une fonction mais on peut l'adapter pour renvoyer : ensemble_des_valeurs, prob '''
    def methode_Noyau(self, agent, K=norm.pdf, nsamples=1000):
        echantillon = draw(agent, nsamples)
        sigma = echantillon.std()
        h = sigma/nsamples**(0.2) # thumb rule
        densite = lambda y : np.sum([K((y-x)/h) for x in echantillon])/(n*h)
        return densite
    
    ''' pour la méthode des différences finies, on retourne une fonction mais on peut l'adapter pour renvoyer : ensemble_des_valeurs, prob '''
    def methode_differences_finies(self, agent, nsamples=1000):
        derivee = (inv_cdf(agent)[1:]-inv_cdf(agent)[:-1])/(nsamples-1)
        ''' regarder s'il y a une méthode plus efficace pour les listes triées dans l'ordre croissant '''
        densite = lambda y : 1/derivee[bisect.bisect(inv_cdf(agent), y)]
        return densite
    
    ''' plutôt dans le cas discret '''
    def methode_fct_repartition(self, agent, nsamples):
        cdf_inv_bis, uniq_inv = np.unique(inv_cdf(agent), unique_inverse=True)
        ensemble_des_valeurs = (cdf_inv_bis[1:] + cdf_inv_bis[;-1])/2
        S = self.pts_discretisation[uniq_inv]
        prob = S[1:]-S[:-1]
        return ensemble_des_valeurs, prob
        
