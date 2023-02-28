import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
from networkx.algorithms import floyd_warshall_predecessor_and_distance
import random

def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: int,
                               directed: bool = False,
                               draw: bool = False):
    """
    Generates a random graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted (in case of undirected graphs)
    """

    
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    edges = combinations(range(num_of_nodes), 2)
    G.add_nodes_from(range(num_of_nodes))
    
    for _, node_edges in groupby(edges, key = lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        if random.random() < 0.5:
            random_edge = random_edge[::-1]
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)
                
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(0, 20)

    if draw: 
        plt.figure(figsize=(10,6))
        if directed:
            # draw with edge weights
            pos = nx.arf_layout(G)
            nx.draw(G,pos, node_color='lightblue', 
                    with_labels=True,
                    node_size=500, 
                    arrowsize=20, 
                    arrows=True)
            labels = nx.get_edge_attributes(G,'weight')
            nx.draw_networkx_edge_labels(G, pos,edge_labels=labels)
            
        else:
            nx.draw(G, node_color='lightblue', 
                with_labels=True, 
                node_size=500)
        
    return G

G =  gnp_random_connected_graph(5, 0.5, directed=False, draw=False)


def prim_algorithm(graph, weights):
    """
    Prim's algorithm for finding the minimum spanning tree of a graph

    Parameters
    ----------
    graph : networkx graph
        Graph to find the minimum spanning tree of
    weights : dict
        Dictionary of edges and their weights
    
    Returns
    -------
    min_edges : list
        List of edges in the minimum spanning tree of the graph
    """

    start_edges = {i:weights[i] for i in graph.edges() if i[0] == 0}
    first_edge = list(min(start_edges, key=start_edges.get))
    min_edges = [(first_edge[0], first_edge[1], {'weight': weights[tuple(first_edge)]})]

    weights.pop((first_edge[0], first_edge[1]))

    used = [min_edges[0][0], min_edges[0][1]]
    unused = [i for i in range(1, len(graph.nodes()))]
    unused.remove(min_edges[0][1])

    sorted_weights_by_value = sorted(weights.items(), key=lambda x: x[1])
    sorted_weights_by_value = [i[0] for i in sorted_weights_by_value]
    index = 0

    while len(min_edges) < len(graph.nodes()) - 1:
        minimum_weight = sorted_weights_by_value[index]

        if minimum_weight[0] in used and minimum_weight[1] in unused:
            to_append = (minimum_weight[0], minimum_weight[1], {'weight': weights[minimum_weight]})
            min_edges.append(to_append)
            used.append(minimum_weight[1])
            used.append(minimum_weight[0])
            unused.remove(minimum_weight[1])

            minimum_weight_index = sorted_weights_by_value.index(minimum_weight)
            sorted_weights_by_value.pop(minimum_weight_index)
            index = 0

        elif minimum_weight[1] in used and minimum_weight[0] in unused:
            to_append = (minimum_weight[0], minimum_weight[1], {'weight': weights[minimum_weight]})
            min_edges.append(to_append)
            used.append(minimum_weight[0])
            used.append(minimum_weight[1])
            unused.remove(minimum_weight[0])

            minimum_weight_index = sorted_weights_by_value.index(minimum_weight)
            sorted_weights_by_value.pop(minimum_weight_index)
            index = 0

        else:
            index += 1
            if len(sorted_weights_by_value) == 0:
                break

    return min_edges


def kruskal_algorithm(graph, weights):
    """
    Kruskal's algorithm for finding the minimum spanning tree of a graph

    Parameters
    ----------
    graph : networkx graph
        Graph to find the minimum spanning tree of
    weights : dict
        Dictionary of edges and their weights
    
    Returns
    -------
    min_edges : list
        List of edges in the minimum spanning tree of the graph
    """

    sorted_edges = sorted(weights.items(), key=lambda x: x[1])
    num_of_nodes = len(graph.nodes())
    sets = [{i} for i in range(num_of_nodes)]
    min_edges = []
    i = 0
    while len(min_edges) != num_of_nodes - 1:
        element = sorted_edges[i]
        edge = set(element[0])
        is_cycle = [edge.issubset(s) for s in sets]
        if True not in is_cycle:
            min_edges.append((element[0][0], element[0][1], {'weight': element[1]}))

            sets = new_sets(sets, edge)
            sorted_edges.pop(i)
            i=0
        else:
            i+=1

    return min_edges


def new_sets(sets, edge):
    """
    Helper function for kruskal_algorithm

    Parameters
    ----------
    sets : list
        List of sets
    edge : set
        Set of nodes
        
    Returns
    -------
    new_sets : list
        List of sets with updated sets
    """
    for s in sets:
        if s.intersection(edge):
            s.update(edge)

    for s in sets:
        for s2 in sets:
            if s != s2 and s.intersection(s2):
                s2.update(s)
    new_sets = []
    for s in sets:
        if s not in new_sets:
            new_sets.append(s)

    return new_sets


def floyd_warshall_algorithm(graph):
    """
    Floyd-Warshall algorithm for finding the shortest path between all pairs of nodes in a graph.

    Parameters
    ----------
    graph : networkx graph
        Undirected graph

    Returns
    -------
    matrix : list
        List of lists of distances between all pairs of nodes in the graph
    """

    _, dist = floyd_warshall_predecessor_and_distance(graph)


    graph = []
    for _, v in dist.items():
        graph.append(dict(v))

    sorted_graph = []
    for i in graph:
        sorted_graph.append({k: i[k] for k in sorted(i.keys())})


    matrix = [[i for i in dicti.values()] for dicti in sorted_graph]
    n = len(matrix)

    n = len(matrix)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])

    has_negative_cycle = any(matrix[i][i] < 0 for i in range(n))
    if has_negative_cycle:
        return "Negative cost cycle exists"

    return matrix


G =  gnp_random_connected_graph(5, 0.5, directed=False, draw=False)

my_prim = prim_algorithm(G, nx.get_edge_attributes(G, 'weight'))
my_kruskal = kruskal_algorithm(G, nx.get_edge_attributes(G, 'weight'))

G = gnp_random_connected_graph(10, 0.5, True, True)
my_floyd_warshall = floyd_warshall_algorithm(G)
