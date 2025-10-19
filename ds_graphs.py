from fontTools.ttx import process


class GraphNode:
    def __init__(self, val):
        self.val = val
        self.neighbors = []

#directed vs undirected: directed graph edges have a direction associated w them
#: weighted vs unweighted: in a weighted graph, edges have a weight w them
#CYlic vs acylic: cylic contain 1 + cycle

#graph traversal DFS and BFS

#DFS
def dfs(node: GraphNode, visited: Set[GraphNode]):
    visited.add(node)
    process(node)
    for neight in node.neighbors
        if neighbor not in visited
            dfs(neighbor, visited)

#BFS
def bfs(node: GraphNode):
    visited = set()
    queue = deque([node])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            process(node)
            for neighbor in node.neighbors:
                queue.append(neighbor)


