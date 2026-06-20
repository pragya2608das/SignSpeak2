import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 75

self_link = [(i, i) for i in range(num_node)]

inward = [(i, i+1) for i in range(num_node-1)]

outward = [(j, i) for (i, j) in inward]

neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):

        if labeling_mode is None:
            return self.A

        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(
                num_node,
                self_link,
                inward,
                outward
            )
        else:
            raise ValueError()

        return A
