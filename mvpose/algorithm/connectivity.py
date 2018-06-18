import networkx as nx


class ConnectivityGraph:

    def __init__(self, func1, func2, func3):
        """

        :param func1: function for unaries
        :param func2: function for iota
        :param func3: function for lambda
        """
        self.func1 = func1
        self.func2 = func2
        self.func3 = func3
        self.lookup = {}
        self.reverse_lookup = {}
        self.added_joints = set()
        self.current_node_id = 1
        self.G = nx.Graph()

    def add_joints(self, jid, unary):
        """
            add a set of joints
        :param jid:
        :param unary: [ 0.9, 0.xx, ... ]
        :return:
        """
        assert jid not in self.added_joints
        self.added_joints.add(jid)
        func1 = self.func1
        for idx, u in enumerate(unary):
            nid = self.current_node_id
            self.lookup[jid, idx] = nid
            self.reverse_lookup[nid] = (jid, idx)
            self.current_node_id += 1
            self.G.add_node(nid, jid=jid, nid=nid, value=func1(u))

    def add_iota_edge(self, jid, distance):
        """
            add internal edges
        :param jid:
        :param distance: [(a, b, value), ...]
        :return:
        """
        func2 = self.func2
        for a, b, d in distance:
            nid1 = self.lookup[jid, a]
            nid2 = self.lookup[jid, b]
            value = func2(d)
            self.G.add_edge(nid1, nid2, value=value)

    def add_lambda_edge(self, jid1, jid2, As, Bs, Scores):
        """
            add external edges
        :param jid1:
        :param jid2:
        :param As: [a1, a2, ...]
        :param Bs: [b1, b2, ...]
        :param Scores: values
        :return:
        """
        assert jid1 != jid2
        assert len(As) == len(Bs)
        assert len(As) == len(Scores)
        func3 = self.func3
        for a, b, s in zip(As, Bs, Scores):
            nid1 = self.lookup[jid1, a]
            nid2 = self.lookup[jid2, b]
            self.G.add_edge(nid1, nid2, value=func3(s))

    def query(self, Nu, Iota, Lambda):
        """

        :param Nu:
        :param Iota:
        :param Lambda:
        :return:
        """
        return 0
