import os


class NetModule(object):

    def __init__(self, name, connections):
        self.name = name
        self.connections = connections if connections else []

    def add_connection(self, node):
        self.connections.append(node)


class DrugModule(NetModule):

    def __init__(self, name, connections):
        super().__init__(name, connections)


class DiseaseModule(NetModule):

    def __init__(self, name, connections):
        super().__init__(name, connections)
