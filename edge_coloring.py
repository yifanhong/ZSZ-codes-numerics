import networkx as nx
from collections import namedtuple
from typing import List, Set


def _canonicalize_edge(e):
    (u,v) = e[:2] # [:2] in case the graph is weighted
    (u,v) = (u,v) if u < v else (v,u)
    return (u,v) + e[2:]

def _edges_with_keys(G : nx.Graph):
    try:
        return G.edges(keys=True)
    except TypeError:
        return G.edges()

def edge_color_bipartite(bipartite_graph : nx.Graph) -> List[Set[int]]:
    '''Given a bipartite graph, return an optimal edge coloring in time O(|VG||EG|).
    This uses the construction in Konz's proof that all bipartite graphs are class 1.'''

    G = bipartite_graph.to_undirected()

    if not nx.is_bipartite(G):
        raise RuntimeError("Graph must be bipartite")

    if nx.number_of_selfloops(G) > 0:
        raise RuntimeError("Graph must not contain self loops")

    graph_degree = max(map(lambda x: x[1], G.degree()))

    ColorSet = namedtuple('ColorSets', ['vertices', 'edges'])
    colorings = [ColorSet(set(), set()) for _ in range(graph_degree)]

    for edge in _edges_with_keys(G):
        (u, v) = edge[:2]
        u_set = None
        try:
            u_set = next(x for x in colorings if u not in x.vertices and v not in x.vertices)
        except StopIteration: # There does not exist a set that does not contain u and does not contain v
            u_set = next(x for x in colorings if u not in x.vertices)
            v_set = next(x for x in colorings if v not in x.vertices)

            # Reform u_set and v_set s.t. we can add the edge to u_set
            # Compute an edge 2-coloring in the subgraph v_set \cup u_set
            # The new sets are the edge colors of this subgraph


            def filter_edge(u,v,key=None):
                # Reassemble the edge
                edge = (u,v) if key is None else (u,v,key)
                return _canonicalize_edge(edge) in u_set.edges or _canonicalize_edge(edge) in v_set.edges
            
            uv_subgraph = nx.subgraph_view(G,
                filter_node = lambda x: (x in u_set.vertices or x in v_set.vertices),
                filter_edge = filter_edge)

            # Fix edge coloring in u_set and v_set so we can add edge
            # We do this by following the chain of edges incident to v and swapping all the colors in this chain
            # ==v_set== u      v ==u_set== x ==v_set== x ==u_set== ....

            u_to_v_set = set()
            v_to_u_set = set()
            for uv_edge in nx.edge_dfs(uv_subgraph, v):
                if _canonicalize_edge(uv_edge) in u_set.edges:
                    u_to_v_set.add(_canonicalize_edge(uv_edge))
                else:
                    v_to_u_set.add(_canonicalize_edge(uv_edge))

            u_to_v_vertices = set(v for edges in u_to_v_set for v in edges)
            v_to_u_vertices = set(v for edges in v_to_u_set for v in edges)

            u_set.edges.difference_update(u_to_v_set)
            u_set.vertices.difference_update(u_to_v_vertices)
            u_set.edges.update(v_to_u_set)
            u_set.vertices.update(v_to_u_vertices)

            v_set.edges.difference_update(v_to_u_set)
            v_set.vertices.difference_update(v_to_u_vertices)
            v_set.edges.update(u_to_v_set)
            v_set.vertices.update(u_to_v_vertices)

        # Add the original edge
        u_set.vertices.add(u)
        u_set.vertices.add(v)
        u_set.edges.add(_canonicalize_edge(edge))
    
    return [v.edges for v in colorings]

