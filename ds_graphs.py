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
def graph_deep_copy_dfs (node: GraphNode) -> GraphNode:
    if not node:
        return None
    return dfs(node)
def dfs(node: GraphNode, clone_map{}) -> GraphNode:
    if node in clone_map:
        return clone_map[node]
    cloned_node = GraphNode(node.val)
    clone_map[node] = cloned_node
    for neighbor in node.neighbors:
        cloned_neighbor = dfs(neighbor, clone_map)
        cloned_node.neighbors.append(cloned_neighbor)
    return cloned_node

def count_islands(matrix: List[List[int]]) -> int:
    if not matrix:
        return 0
    count = 0
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            if matrix[r][c] == 1:
                dfs(r,c,matrix)
                count += 1
    return count
def dfs(r: int, c: int, matrix: List[List[int]]) -> None:
    matrix[r][c] = -1
    dirs = [(-1, 0), (1, 0), (0, -1),(0, 1)]

    for d in dirs:
        next_r, next_c = r + d[0], c + d[1]
        if(is_within_bounds(next_r, next_c, matrix)
           and matrix[next_r][next_c] == 1):
            dfs(next_r, next_c, matrix)

def is_within_bounds(r: int, c: int, matrix: List[List[int]]) -> bool:
    return 0 <= r < len(matrix) and 0 <= c < len(matrix[0])
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


