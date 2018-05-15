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

        self.nodes_per_joint = [set() for i in range(n_joints)]  # all nodes in a joint group

        G = nx.Graph()
        self.lookup = {}
        self.reverse_lookup = {}
        for idx, (jid, nid) in enumerate(D):
            self.lookup[jid, nid] = idx + 1
            self.reverse_lookup[idx + 1] = (jid, nid)
            G.add_node(idx + 1, jid=jid, nid=nid, color=ncolors[jid])

            self.nodes_per_joint[jid].add(idx+1)

        lambda_down = {}    # lookup for follow-up joints on limbs => a -> b
        lambda_up = {}      # lookup for bubble-up joints on limbs => b -> a
        self.lambda_down = lambda_down
        self.lambda_up = lambda_up

        # ~ joints
        for jid1, jid2, a, b in E_l:
            assert jid1 != jid2

            if jid1 not in lambda_down:
                lambda_down[jid1] = set()
            lambda_down[jid1].add(jid2)
            if jid2 not in lambda_up:
                lambda_up[jid2] = set()
            lambda_up[jid2].add(jid1)

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

    def query_with_choice(self, jid1, a):
        """
            queries all transitivity objectives

            Attention: so far direct cicular references over several
            limbs is not implemented!
        :param jid1:
        :param a:
        :return: intra and inter transitivity lists:
            intra: [ (jid, a, b, c), ...]               ab + ac -1 <= bc
            intra_choice: [ (jid, a, b, c), ... ]       ab or ac
            inter: [ (jid1, a, jid2, b, jid3, c), ...]  ab + ac -1 <= bc
            inter_choice: [(jid1, a, jid2, b, jid3, c), ...]
        """
        G = self.G
        node_a = self.lookup[jid1, a]
        N_a = frozenset(G.neighbors(node_a))

        print("a=", node_a)
        print('\tN_a:', N_a)

        # transitivity constraints:
        # a--b--c   => if a--b AND a--c ALSO b--c
        #           => if a--b AND b--c ALSO a--c
        #           => if a--c AND b--c ALSO a--b
        intra = []
        inter = []

        # choice constraints:
        # a--b--c   => if a--b AND a--c BUT NOT b--c CHOOSE a--b XOR a--c
        #           => if a--b AND b--c BUT NOT a--c CHOOSE a--b XOR b--c
        #           => if a--c AND b--c BUT NOT a--b CHOOSE a--c XOR b--c
        intra_choice = []
        inter_choice = []

        for node_b in self.nodes_per_joint[jid1]:

            # ====================
            # intra-joint handling
            # ====================
            if node_b == node_a:
                continue
            jid2, b = self.reverse_lookup[node_b]
            assert jid2 == jid1
            if b < a:
                continue

            N_b = frozenset(G.neighbors(node_b))
            print("\tb=", node_b)
            print('\t\tN_b:', N_b)

            ab = node_b in N_a
            assert ab == (node_a in N_b)

            for node_c in self.nodes_per_joint[jid1]:
                if node_c == node_b:
                    continue
                jid3, c = self.reverse_lookup[node_c]
                assert jid3 == jid1
                if c < b:
                    continue

                ac = node_c in N_a
                bc = node_c in N_b

                if ab:
                    if ac and bc:                      # ab + ac -1 <= bc
                        intra.append((jid1, a, b, c))  # ensure transitivity
                        intra.append((jid1, b, a, c))  # add all 3 combinations
                        intra.append((jid1, c, a, b))
                    elif ac:
                        intra_choice.append((jid1, a, b, c))  # ab or ac
                    elif bc:
                        intra_choice.append((jid1, b, a, c))  # ab or bc
                elif ac and bc:
                    assert node_a not in N_b
                    intra_choice.append((jid1, c, a, b))  # ac or bc

            # ========================
            # inter-joint handling (1)
            # ========================
            # handle [a]--[b]
            #         |  /
            #        (c)
            if jid1 in self.lambda_down:
                for jid3 in self.lambda_down[jid1]:
                    assert jid3 != jid1
                    for node_c in self.nodes_per_joint[jid3]:
                        jid3_, c = self.reverse_lookup[node_c]
                        assert jid3_ == jid3

                        ac = node_c in N_a
                        bc = node_c in N_b

                        # print('handle:', (node_a, node_b, node_c))
                        # print('\tab,ac,bc', (ab, ac, bc))
                        # print('\tNb:', N_b)

                        if ab:
                            if ac and bc:  # ab + ac -1 <= bc
                                inter.append((jid1, a, jid2, b, jid3, c))  # ensure transitivity
                                inter.append((jid2, b, jid1, a, jid3, c))
                                inter.append((jid3, c, jid1, a, jid2, b))
                            elif ac:
                                inter_choice.append((jid1, a, jid2, b, jid3, c))  # ab or ac
                            elif bc:
                                inter_choice.append((jid2, b, jid1, a, jid3, c))  # ab or bc
                        elif ac and bc:
                            assert node_a not in N_b
                            inter_choice.append((jid3, c, jid1, a, jid2, b))  # ac or bc

        # ========================
        # inter-joint handling (2)
        # ========================
        # handle [a]
        #         |  \
        #        (b)--(c)
        if jid1 in self.lambda_down:
            for jid2 in self.lambda_down[jid1]:
                assert jid1 != jid2
                for node_b in self.nodes_per_joint[jid2]:
                    jid2_, b = self.reverse_lookup[node_b]
                    assert jid2_ == jid2

                    N_b = frozenset(G.neighbors(node_b))
                    print("\tb=", node_b)
                    print('\t\tN_b:', N_b)

                    ab = node_b in N_a
                    assert node_a in N_b

                    for node_c in self.nodes_per_joint[jid2]:
                        jid3, c = self.reverse_lookup[node_c]
                        assert jid3 == jid2
                        if b < c:
                            continue

                        N_c = frozenset(G.neighbors(node_c))

                        ac = node_c in N_a
                        bc = node_c in N_b

                        if ab:
                            if ac and bc:  # ab + ac -1 <= bc
                                inter.append((jid1, a, jid2, b, jid3, c))  # ensure transitivity
                                inter.append((jid2, b, jid1, a, jid3, c))
                                inter.append((jid3, c, jid1, a, jid2, b))
                            elif ac:
                                inter_choice.append((jid1, a, jid2, b, jid3, c))  # ab or ac
                            elif bc:
                                inter_choice.append((jid2, b, jid1, a, jid3, c))  # ab or bc
                        elif ac and bc:
                            assert node_a not in N_b
                            inter_choice.append((jid3, c, jid1, a, jid2, b))  # ac or bc

        return intra, intra_choice, inter, inter_choice

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
