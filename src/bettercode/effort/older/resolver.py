class DependencyResolver:
    def __init__(self):
        self.adj = {}
        self._memo = {}

    def add_connection(self, u, v):
        if u not in self.adj: self.adj[u] = []
        self.adj[u].append(v)

    def find_longest_chain(self, start_node):
        return self._dfs(start_node, set())

    def _dfs(self, u, visited):
        if u in self._memo:
            return self._memo[u]

        max_len = 0
        visited.add(u)
        
        if u in self.adj:
            for v in self.adj[u]:
                if v not in visited:
                    dist = 1 + self._dfs(v, visited)
                    if dist > max_len:
                        max_len = dist
        
        visited.remove(u)
        self._memo[u] = max_len
        return max_len
