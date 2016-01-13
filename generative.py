import brain_graphs
from igraph import Graph, VertexClustering
import numpy as np
import pandas as pd
from multiprocessing import Pool
from itertools import combinations

def calculate_node_within_community_betweeness(graph,membership,community):
	wcsp = np.zeros((graph.vcount()))
	community_nodes = np.where(membership==community)[0]
	for node1,node2 in combinations(community_nodes,2):
		shortest_path = graph.get_shortest_paths(node1,node2,weights='weight',output ='vpath')[0][1:-1]
		wcsp[shortest_path] = wcsp[shortest_path] + 1
	return wcsp

def calculate_node_between_community_betweeness(graph,membership,community):
	bcsp = np.zeros((graph.vcount()))
	for node1,node2 in combinations(range(len(bcsp)),2):
		if membership[node1] == membership[node2]:
			continue
		shortest_path = graph.get_shortest_paths(node1,node2,weights='weight',output ='vpath')[0][1:-1]
		bcsp[shortest_path] = bcsp[shortest_path] + 1
	bcsp[membership==community] = 0.0
	return bcsp

def generate_random_edges(graph,node,n_edges,possible_nodes,membership,within):
	nodes = []
	for n in possible_nodes:
		if n == node:
			continue
		if membership[n]!=membership[node] and within == False:
			nodes.append(n)
		if membership[n]==membership[node] and within == True:
			nodes.append(n)
	assert len(nodes) >= n_edges, 'Not enough nodes in community to create that many edges'
	edges = []
	while len(edges) < n_edges:
		edge = (node,np.random.choice(nodes))
		if edge[0] == edge[1]:
			continue
		if graph.get_eid(edge[0],edge[1],error=False) >= 0:
			continue
		if edge in edges:
			continue
		if (edge[1],edge[0]) in edges:
			continue
		edges.append(edge)
	return edges

def add_edges(graph,edges):
	for edge in edges:
		if graph.get_eid(edge[0],edge[1],error=False) >= 0:
			print 'Edge already in graph'
			break
		else:
		    graph.add_edge(edge[0],edge[1], weight=1)
	return graph

def calculate_n_nodes_to_add(density,gamma,n_nodes):
	n_total_edges = (n_nodes*(n_nodes-1.))/2.
	# number of total edges in intial graph at density we select 
	n_edges = n_total_edges * density
	# number of within community edges 
	n_within_edges = n_edges * gamma
	# number of between community edges
	n_between_edges = n_edges - n_within_edges
	return n_between_edges,n_within_edges

def initial_graph(n_nodes,seed_nodes,membership,gamma=.8,density=.15):
	# make inital empty graph
	graph = Graph(directed=False)
	# add all nodes
	graph.add_vertices(range(n_nodes))
	# given number of nodes in intial graph, calculate total possible edges
	n_between_edges, n_within_edges = calculate_n_nodes_to_add(density,gamma,seed_nodes)
	# randomly sample membership distribution to add nodes to intial graph
	nodes = []
	# however, ensure at least 2 nodes in each community
	for community in np.unique(membership):
		community_nodes = np.array(np.where(membership==community)[0])
		while True:
			new_nodes = np.random.choice(community_nodes,2)
			if new_nodes[0] != new_nodes[1]:
				break
		for node in new_nodes:
			nodes.append(node)
	while len(nodes) < seed_nodes:
		community = np.random.choice(membership)
		community_nodes = np.where(membership==community)[0]
		new_node = np.random.choice(community_nodes)
		if new_node in nodes:
			continue
		nodes.append(new_node)
	assert len(np.unique(nodes)) == seed_nodes
	# randomly pick number of edges to add to graph, fitting for density
	# however, give every node at least 1 within community edge
	between_edge_numbers = dict(zip(nodes,np.zeros((seed_nodes))))
	within_edge_numbers = dict(zip(nodes,np.zeros((seed_nodes))))
	while np.sum(within_edge_numbers.values()) < n_within_edges:
		index = np.random.choice(nodes)
		within_edge_numbers[index] = within_edge_numbers[index] + 1
	while np.sum(between_edge_numbers.values()) < n_between_edges:
		index = np.random.choice(nodes)
		between_edge_numbers[index] = between_edge_numbers[index] + 1
	for node in nodes:
		within_edges = generate_random_edges(graph=graph,node=node,n_edges=within_edge_numbers[node],possible_nodes=nodes,membership=membership,within=True)
		graph = add_edges(graph,within_edges)
		between_edges = generate_random_edges(graph=graph,node=node,n_edges=between_edge_numbers[node],possible_nodes=nodes,membership=membership,within=False)
		graph = add_edges(graph,between_edges)
	return graph

def find_edgeless_nodes(graph):
	edgeless_nodes = np.array(graph.degree())
	edged_nodes = np.where(edgeless_nodes>0)[0]
	edgeless_nodes = np.where(edgeless_nodes==0)[0]
	return edgeless_nodes,edged_nodes

"""
Parameters

module_sizes: list of size of each module
gamma: fraction of between and within module edges
density: fraction of edges that exists relative to complete graph
n_nodes: number of nodes to add to graph
seed_nodes: number of nodes to initialize graph

Initialization

Assign seed_nodes to modules, according to size, and randomly add edges to match density and gamma

Generative Growth Model 

1. Compute number of between and within module edges to add, based on density and fraction

within module edge
for each node, 1 / average shortest path. to power of BETA
	add edges to nodes with most shortest paths through it.

between module edge
for each node not in the module, get average shortest path between nodes not in module, but consider paths through module. 
	add edges to nodes with most shortest paths through it. 
"""

n_nodes = 100
n_communities = 2
seed_nodes = 10
gamma = .5
density = .05
membership = []
for n in range(n_nodes):
	membership.append(np.random.randint(0,n_communities))
membership = np.array(membership)
graph = initial_graph(n_nodes,seed_nodes,membership,gamma,density)
n_nodes_in_graph = seed_nodes
while n_nodes_in_graph < n_nodes:
	#find nodes that have not been added to the graph.
	edgeless_nodes,edged_nodes = find_edgeless_nodes(graph)
	community = np.random.choice(membership)
	nodes = np.where(membership==community)[0]
	nodes = np.delete(nodes,edged_nodes)
	node = np.random.choice(nodes)
	community = membership[node]
	wcb = calculate_node_within_community_betweeness(graph,membership,community)
	bcb = calculate_node_between_community_betweeness(graph,membership,community)
	n_between_edges, n_within_edges = calculate_n_nodes_to_add(density,gamma,n_nodes_in_graph)
	sorted_bcb = np.argsort(bcb)
	sorted_wcb = np.argsort(wcb)
	within_nodes = sorted_wcb[-int(n_within_edges):]
	between_nodes = sorted_bcb[-int(n_between_edges):]
	#need to find if nodes are already connected before passing.
	#or just pass and add the ones in order. 
	within_edges = generate_random_edges(graph=graph,node=node,n_edges=int(n_within_edges),possible_nodes=within_nodes,membership=membership,within=True)
	graph = add_edges(graph,within_edges)
	between_edges = generate_random_edges(graph=graph,node=node,n_edges=int(n_between_edges),possible_nodes=between_nodes,membership=membership,within=False)
	graph = add_edges(graph,between_edges)
	n_nodes_in_graph = len(find_edgeless_nodes(graph)[1])
	1/0


