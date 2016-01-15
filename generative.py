import brain_graphs
from igraph import Graph, VertexClustering
import numpy as np
import pandas as pd
from multiprocessing import Pool
from itertools import combinations
import pylab as plt
import seaborn as sns
from multiprocessing import Pool
import glob

def get_power_pc(hub='pc'):
    data = pd.read_csv('/home/despoB/mb3152/modularity/mmc3.csv')
    data['new'] = np.zeros(len(data))
    if hub == 'pc':
        for x2,y2,z2,pc in zip(data.X2,data.Y2,data.Z2,data.PC):
            data.new[data[data.X1==x2][data.Z1==z2][data.Y1==y2].index[0]] = pc
    else:
    	1/0
        for roi_num,x,y,z,ap in zip(data.index,data.X1,data.Y1,data.Z1,data.CD):
            fill_index = data.new[data.X2==x][data.Y2==y][data.Z2==z]
            data.new[fill_index.index.values[0]] = ap
    values_dict = data.new.to_dict()
    return values_dict

def calculate_node_within_community_betweeness(graph,membership,community):
	wcsp = np.zeros((graph.vcount()))
	community_nodes = np.where(membership==community)[0]
	for node1,node2 in combinations(community_nodes,2):
		shortest_path = graph.get_shortest_paths(node1,node2,weights='weight',output ='vpath')[0][1:-1]
		wcsp[shortest_path] = wcsp[shortest_path] + 1
	return wcsp

def calculate_node_between_community_betweeness(graph,nodes,membership,community):
	bcsp = np.zeros((graph.vcount()))
	for node1,node2 in combinations(nodes,2):
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
		if graph.get_eid(node,n,error=False) != -1:
			continue
		if membership[n]!=membership[node] and within == False:
			nodes.append(n)
		if membership[n]==membership[node] and within == True:
			nodes.append(n)
	for n in nodes:
		if within == False:
			assert membership[n] != membership[node]
		else:
			assert membership[n] == membership[node]
	assert len(nodes) >= n_edges, 'Not enough nodes in community to create that many edges'
	while len(edges) < n_edges:
		edge = (node,np.random.choice(nodes))
		if edge[0] == edge[1]:
			continue
		if edge in edges:
			continue
		if (edge[1],edge[0]) in edges:
			continue
		edges.append(edge)
	return edges

def generate_and_add_between_edges(graph,membership,node,bcb,n_between_edges):
	sorted_bcb = np.argsort(bcb)
	sorted_bcb = np.fliplr([sorted_bcb])[0]
	nba = 0.
	while True:
		for node2 in sorted_bcb:
			if membership[node] == membership[node2]:
				continue
			if bcb[node2] == 0.0:
				node2 = np.random.choice(np.where(bcb==0.0)[0].reshape(-1))
			if graph.get_eid(node,node2,error=False) != -1:
				continue
			if graph.degree()[node2] == 0.0:
				continue
			graph = add_edges(graph,[[node,node2]])
			nba = nba + 1
			if nba >= n_between_edges:
				break
		if nba >= n_between_edges:
			break
	return graph

def generate_and_add_within_edges(graph,membership,node,wcb,n_within_edges):
	sorted_wcb = np.argsort(wcb)
	sorted_wcb = np.fliplr([sorted_wcb])[0]
	nwa = 0.
	while True:
		for node2 in sorted_wcb:
			if membership[node] != membership[node2]:
				continue
			if wcb[node2] == 0.0:
				node2 = np.random.choice(np.where(wcb==0.0)[0].reshape(-1))
			if graph.get_eid(node,node2,error=False) != -1:
				continue
			if graph.degree()[node2] == 0.0:
				continue
			graph = add_edges(graph,[[node,node2]])
			nwa = nwa + 1
			if nwa >= n_within_edges:
				break
		if nwa >= n_within_edges:
			break
	return graph

def add_edges(graph,edges):
	for edge in edges:
		try:
			assert graph.get_eid(edge[0],edge[1],error=False) == -1
			graph.add_edge(edge[0],edge[1],weight=1)
		except:
			1/0
	return graph

# def calculate_num_edges(density,gamma,n_nodes):
# 	n_total_edges = (n_nodes*(n_nodes-1.))/2.
# 	# number of total edges in intial graph at density we select 
# 	n_edges = n_total_edges * density
# 	# number of within community edges 
# 	n_within_edges = n_edges * gamma
# 	# number of between community edges
# 	n_between_edges = n_edges - n_within_edges
# 	return n_between_edges,n_within_edges

def calculate_num_edges(density,gammas,n_nodes):
	n_total_edges = (n_nodes*(n_nodes-1.))/2.
	# number of total edges in intial graph at density we select 
	n_edges = int(np.ceil(n_total_edges * density))
	n_between_edges = 0
	n_within_edges = 0
	g = gammas[np.random.randint(0,len(gammas))]
	assert sum(g) > 0.0
	pr = g/sum(g)
	for n in range(n_edges):
		choice = np.random.choice([0,1],p=pr)
		if choice == 0:
			n_within_edges += 1
		else:
			n_between_edges += 1
	return n_between_edges,n_within_edges

def initial_graph(seed_nodes,n_nodes,n_communities,pi,gamma=.8,density=.15):
	#make node array for seeds
	nodes = np.array(range(seed_nodes))
	#make membership
	membership = np.zeros(n_nodes)-1
	# make inital empty graph
	graph = Graph(directed=False)
	# add all nodes
	graph.add_vertices(range(len(membership)))
	# given number of nodes in intial graph, calculate total possible edges
	n_between_edges, n_within_edges = calculate_num_edges(density=density,gamma=gamma,n_nodes=seed_nodes)
	n_between_edges = np.ceil(np.ceil(n_between_edges)/seed_nodes)
	n_within_edges = np.floor(np.floor(n_within_edges)/seed_nodes)
	print 'Initializing with %s nodes, %s within community edges, and %s between community edges' %(seed_nodes,n_within_edges,n_between_edges)
	# initialize membership
	curr_length = 0
	for i in range(n_communities):
		num_nodes = np.ceil(pi[i]*seed_nodes)
		membership[curr_length:curr_length+num_nodes] = i
		curr_length += num_nodes
	# add in edges
	for node in nodes:
		within_edges = generate_random_edges(graph=graph,node=node,n_edges=n_within_edges,possible_nodes=nodes,membership=membership,within=True)
		graph = add_edges(graph,within_edges)
		between_edges = generate_random_edges(graph=graph,node=node,n_edges=n_between_edges,possible_nodes=nodes,membership=membership,within=False)
		graph = add_edges(graph,between_edges)
	return graph,membership

def find_edgeless_nodes(graph):
	edgeless_nodes = np.array(graph.degree())
	edged_nodes = np.where(edgeless_nodes>0)[0]
	edgeless_nodes = np.where(edgeless_nodes==0)[0]
	return edgeless_nodes,edged_nodes

def find_new_node(graph,pi,n_communities):
	found = False
	while found == False:
		edgeless_nodes,edged_nodes = find_edgeless_nodes(graph)
		node = np.random.choice(edgeless_nodes)
		try:
			assert node not in edged_nodes
			found = True
		except:
			found = False
	return node,np.random.choice(range(n_communities),p=pi)

def get_gammas(density=.2):
	gammas = []
	matrices = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/*rfMRI_REST*matrix*')
	matrix = np.load(matrices[0])
	for m in matrices[1:]:
		matrix = np.nansum([matrix,np.load(m)],axis=0)
	matrix = matrix/len(matrices)
	# matrix = brain_graphs.threshold(matrix,density,binary=False,check_tri=True,interpolation='midpoint',normalize=False)
	graph = brain_graphs.matrix_to_igraph(matrix,density,binary=False,check_tri=True,interpolation='midpoint',normalize=False)
	graph = graph.community_infomap(edge_weights='weight')
	membership = np.array(graph.membership)
	for node in range(matrix.shape[0]):
		community = membership[node]
		community_nodes = np.argwhere(membership==community)
		non_community_nodes = np.argwhere(membership!=community)
		within = np.ceil(np.sum(matrix[node,community_nodes]))
		between = np.ceil(np.sum(matrix[node,non_community_nodes]))
		if within + between == 0.0:
			continue
		gammas.append([within,between])
	return gammas

1/0
# def generative_model_shortest_paths_win(x,n_nodes=264,n_communities=4,gamma=.95,density=.1,intial_community_density=.1,equal_community_size=False):

n_nodes=264
n_communities=4
density=.1
intial_community_density=.1
equal_community_size=False

"""
Parameters
n_nodes: number of nodes in final iteration of the graph
n_communities: number of communities in graph
gamma: fraction of between and within module edges
gamma, and its distribution should be extimated on real data. Just get a set of ratios at given density, and randomly choose from it. 
density: fraction of edges that exists relative to complete graph
intial_community_density: the density of nodes that are added to each community in initialization of graph
equal_community_size: True, or False if uneven communities are wanted, for which a dirichlet distribution is used.

Initialization

Caclulate the number of nodes to modules, according to size, and randomly add edges to match density and gamma

Generative Growth Model

1. 
Compute number of between and within module edges to add, based on density and fraction
2. 
within module edge
for each node, 1 / average shortest path. to power of BETA
	add edges to nodes with most shortest paths through it.

between module edge
for each node not in the module, get average shortest path between nodes not in module, but consider paths through module. 
	add edges to nodes with most shortest paths through it. 
"""
#intialize graph with number of nodes based on community density, total n_nodes, and n_communities.
#this ensures that a certain percentage of nodes in each community are present in initalization of graph.
#for example, an intial_community_density of .2 ensure that each community has twenty percent of its nodes present during initialization
seed_nodes = np.ceil(intial_community_density *(n_nodes/n_communities))*n_communities
gamma = get_gammas(density)
#caclulate the number of within community edges that each node will have.
#this is important, as we want to make sure we can actually create a graph with the supplied parameters
#for example, if the gamma and density is too high, and communities too small, it is impossible to initialize this graph.
n_within_edges = np.ceil(calculate_num_edges(density=density,gamma=gammas,n_nodes=seed_nodes)[1])
n_within_edges_per_node = np.ceil(n_within_edges/seed_nodes)
#generate community sizes by making a distribution
if equal_community_size:
	pi = np.ones(n_communities)/n_communities
else:
	for i in range(10000):
		pi = np.random.dirichlet(alpha=np.ones(n_communities))
		#make sure all communities have enough nodes to satisfy gamma
		large_enough = 0
		for i in range(len(pi)):
			#calculate the size of the community, assuming it might get rounded down.
			community_size = np.floor(pi[i]*seed_nodes)
			#calculate the number of edges this community can have total
			possible_within_edges = np.floor((community_size*(community_size-1))/2.)
			#ensure that we do not have more within_community edges than possible for community
			if n_within_edges_per_node * (community_size+1) < possible_within_edges:
				large_enough += 1
		if large_enough == n_communities:
			break
	assert i < 9999, 'Could not satisfy parameters'
#pass pi and parameters to make graph.
graph,membership = initial_graph(int(seed_nodes),int(n_nodes),int(n_communities),pi,gamma,density)
n_nodes_in_graph = len(find_edgeless_nodes(graph)[1])
assert n_nodes_in_graph == seed_nodes, 'Initialization failed, different number of nodes than requested'
print 'Graph initialized with ' + str(n_nodes_in_graph) + ' nodes and a density of: ' + str(graph.ecount()/((n_nodes_in_graph*(n_nodes_in_graph-1))/2.))
while n_nodes_in_graph < n_nodes:
	if str(n_nodes_in_graph)[-1] == '0':
		print str(n_nodes_in_graph) + str(' nodes, ') + str(graph.ecount()) + ' edges in graph and density of: ' + str(graph.ecount()/((n_nodes_in_graph*(n_nodes_in_graph-1))/2.))
	#find nodes that have not been added to the graph.
	node,community = find_new_node(graph,pi,n_communities)
	membership[node] = community
	wcb = calculate_node_within_community_betweeness(graph,membership,community)
	bcb = calculate_node_between_community_betweeness(graph,np.argwhere(membership!=-1).reshape(-1),membership,community)
	n_between_edges, n_within_edges = calculate_num_edges(density,gamma,n_nodes_in_graph+1)
	n_between_edges = np.floor(np.ceil(n_between_edges)/(n_nodes_in_graph+1))
	n_within_edges = np.ceil(np.ceil(n_within_edges)/(n_nodes_in_graph+1))
	if str(n_nodes_in_graph)[-1] == '0':
		print 'adding %s within community edges and %s between community edges' %(n_within_edges,n_between_edges)
	#could select within based on both wcb and bcb, so that you get connector "hubs" with high PC and WMDZ
	generate_and_add_between_edges(graph,membership,node,bcb+wcb,n_between_edges)
	generate_and_add_within_edges(graph,membership,node,bcb+wcb,n_within_edges)
	n_nodes_in_graph = len(find_edgeless_nodes(graph)[1])
brain_graph = brain_graphs.brain_graph(VertexClustering(graph,membership=membership.astype(int)))


# def main(n_trials=20,n_cores=20):
# pool = Pool(n_cores)
# return pool.map(generative_model_shortest_paths_win,np.zeros(n_trials)) 

# brains = main()






















