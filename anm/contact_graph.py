from scipy.spatial.distance import pdist as pd
from scipy.spatial.distance import squareform as sf
import numpy as np
import networkx as nx

def distance_matrix(coords):
    return sf(pd(coords))

def build_contact_graph(coords, cutoff):
    D = distance_matrix(coords)
    adj = (D<=cutoff) & (D>0)
    contacts = np.argwhere(np.triu(adj, k=1))
    return adj, contacts, D

                           
def check_connected(adj):
    G = nx.from_numpy_array(adj)
    return nx.is_connected(G), list(nx.connected_components(G))