import os


class NetModule(object):

    def __init__(self, name, connected_nodes, weighted=False):
        """A group of nodes connected to a given concept.

        Arguments:
            name {str} -- the name of the concept
            connections {iterable} -- a group of nodes

        Keyword Arguments:
            weighted {bool} -- Whther w or w/o connection weights (default: {False})
        """
        self.name = name
        self.connections = []
        if connected_nodes:
            for node in connected_nodes:
                self.add_connection(node)

    def add_connection(self, node):
        self.connections.append(node)


class DrugModule(NetModule):

    def __init__(self, name, connected_nodes):
        super().__init__(name, connected_nodes)


class DiseaseModule(NetModule):

    def __init__(self, name, connected_nodes):
        super().__init__(name, connected_nodes)
