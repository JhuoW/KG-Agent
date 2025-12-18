import networkx as nx
from collections import deque


def build_graph(graph: list, undirected = False) -> nx.DiGraph | nx.Graph:
    if undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h.strip(), t.strip(), relation=r.strip())
    return G


def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    """
    Get shortest paths connecting question and answer entities.
    """
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p) - 1):
            u = p[i]
            v = p[i + 1]
            tmp.append((u, graph[u][v]["relation"], v))
        result_paths.append(tmp)
    return result_paths