class Datasource(object):
    def __init__(self, name: str, connection_string: str, query: str, cachable: bool):
        self.name = name
        self.connection_string = connection_string
        self.query = query
        self.cachable = cachable

