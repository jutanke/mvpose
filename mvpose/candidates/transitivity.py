import networkx as nx
from cselect import color as cs
import numpy as np


class TransitivityLookup:
    """
        This tool helps to set the transitivity rules for the optimization graph cut
    """

    def __init__(self, D, E_l, E_j):
        """

        :param D: set of all nodes: [ (jid, nid), ... ]
        :param E_l: set of all limb edges: [ (jid1, jid2, a, b), ... ]
        :param E_j: set of all joint edges: [ (jid, a, b), ... ]
        """
        if not type(D) == np.ndarray:
            D = np.array(D)

        n_joints = np.max(np.array(D)[:,0]) + 1
        ncolors = cs.lincolor(n_joints +2)/255

        # set the maximal value that an index per joint can yield
        count_nodes_per_joint = [0] * n_joints
        for jid in range(n_joints):
            M = D[:, 0] == jid
            if np.sum(M) > 0:
                max_id = np.max(D[M][:,1])
                count_nodes_per_joint[jid] = max_id + 1
                assert np.sum(M) == count_nodes_per_joint[jid]
        self.count_nodes_per_joint = count_nodes_per_joint

        self.nodes_per_joint = [set() for i in range(n_joints)]  # all nodes in a joint group

        G = nx.Graph()
        self.lookup = {}
        self.reverse_lookup = {}
        for idx, (jid, nid) in enumerate(D):
            self.lookup[jid, nid] = idx + 1
            self.reverse_lookup[idx + 1] = (jid, nid)
            G.add_node(idx + 1, jid=jid, nid=nid, color=ncolors[jid])

            self.nodes_per_joint[jid].add(idx+1)

        # ~ joints
        for jid1, jid2, a, b in E_l:
            assert jid1 != jid2
            n1 = self.lookup[jid1, a]
            n2 = self.lookup[jid2, b]
            G.add_edge(n1, n2, color='teal')

        for jid, a, b in E_j:
            assert a != b
            n1 = self.lookup[jid, a]
            n2 = self.lookup[jid, b]
            G.add_edge(n1, n2, color='orange')

        self.G = G

    def plot(self):
        G = self.G
        edges = G.edges()
        ecolors = [G[u][v]['color'] for u, v in edges]
        nodes = G.nodes()
        ncolors = [nodes[u]['color'] for u in nodes]
        nx.draw(G, edge_color=ecolors, node_color=ncolors, with_labels=True)

    def find_invalid(self, jid, a):
        """
            As the limbs are fully connected in theory and only sparse to
            save memory we need to query all invalid connections so that
            we can make sure that this ones are excluded in the optimization
        :param jid:
        :param a:
        :return:
        """
        pass

    def query(self, jid1, a):
        """
            queries all transitivity objectives
        :param jid1:
        :param a:
        :return: intra and inter transitivity lists:
            intra: [ (jid, a, b, c), ...]
            inter: [ (jid1, a, jid2 b, jid3, c), ...]
        """
        G = self.G

        n_a = self.lookup[jid1, a]

        N = frozenset(G.neighbors(n_a))

        intra = []
        inter = []

        inter_already_handled = set()

        for n_b in N:
            jid2, b = self.reverse_lookup[n_b]
            n_neighbor = frozenset(G.neighbors(n_b))
            intersection = N.intersection(n_neighbor)

            for n_c in intersection:
                jid3, c = self.reverse_lookup[n_c]

                c1 = jid1 == jid2
                c2 = jid1 == jid3
                c3 = jid2 == jid3

                if c1 and c3:
                    # a < c must be true so that we ignore symmetric cases:
                    #
                    #  * * * * * *
                    #  *         *
                    # [a]--[b]--[c]  ==> (a, b) -> (c)
                    #                    (c, b) -> (a)
                    if a < b < c:
                        intra.append(
                            (jid1, a, b, c)
                        )
                elif c1:
                    if a < b and jid1 < jid3 and not (n_a, n_b, n_c) in inter_already_handled:
                        inter.append((jid1, a, b, jid3, c))
                        inter_already_handled.add((n_a, n_b, n_c))
                elif c2:
                    if a < c and jid1 < jid2 and not (n_a, n_c, n_b) in inter_already_handled:
                        inter.append((jid1, a, c, jid2, b))
                        inter_already_handled.add((n_a, n_c, n_b))
                elif c3:
                    pass  # this is being handled in another setup
                    # if b < c and not (n_b, n_c, n_a) in inter_already_handled:
                    #     inter.append((jid2, b, c, jid1, a))
                    #     inter_already_handled.add((n_b, n_c, n_a))
                else:  # circle
                    pass
                    # if jid1 < jid2 < jid3 and not (n_a, n_c, n_b) in inter_already_handled:
                    #     inter.append((jid1, a, jid2, b, jid3, c))
                    #     inter_already_handled.add((n_a, n_c, n_b))
                    #raise ValueError("jids:", (jid1, jid2, jid3))

        return intra, inter
