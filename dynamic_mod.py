#!/home/despoB/mb3152/anaconda/bin/python
import brain_graphs
import pandas as pd
import os
import sys
import scipy.io as sio
from scipy.stats.stats import pearsonr
from scipy.stats import ttest_ind
import numpy as np
import networkx as nx
import scipy.stats
import subprocess
import pickle
import scipy
import h5py
import random
from scipy.io import loadmat
import nibabel as nib
from sklearn.metrics.cluster import normalized_mutual_info_score
from itertools import combinations
from igraph import Graph, ADJ_UNDIRECTED
import warnings
import logging
from scipy.stats import ttest_ind
import glob
# logging.captureWarnings(True)
# warnings.catch_warnings(record=True)
import math
from collections import Counter
import matplotlib.pylab as plt
plt.rcParams['pdf.fonttype'] = 42
import seaborn as sns
from scipy.stats.mstats import zscore as z_score
from igraph import VertexClustering
import powerlaw
#build graphs for timepoints when component is engaged.
#build graphs for when no variance versus high variance, look at modularity and PC and WMD. Perhaps calculate 
# modularity without PC nodes / See if most of the between module connections come from PC nodes.

project = 'nki'
data_dir = '/home/despoB/mb3152/data/nki_data/preprocessed/pipeline_comp_cor_and_standard'
subject_dir = '%s/SUBJECT_session_1/functional_mni/_scan_RfMRI_mx_645_rest/_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96/_compcor_ncomponents_5_selector_pc10.linear1.wm0.global0.motion1.quadratic1.gm0.compcor1.csf0/_bandpass_freqs_0.009.0.08/**' %(data_dir)

subjects = ['0194023', '0185428', '0123657', '0141795', '0163508', '0123971', '0158411', '0185781', '0103714',
 '0174363', '0188854', '0136303', '0144667', '0139480', '0163228', '0154423', '0187635', '0179005', '0154555',
 '0150404', '0168357', '0192197', '0137496', '0159429', '0142673', '0180093', '0141860', '0116065', '0134795',
 '0150525', '0196198', '0162704', '0193358', '0105290', '0137073', '0112249', '0117168', '0125747', '0119947',
 '0138333', '0114688', '0167693', '0127665', '0117747', '0113436', '0127484', '0172267', '0152992', '0197584',
 '0176913', '0176479', '0188324', '0181179', '0159461', '0188757', '0102826', '0170400', '0112586', '0161348',
 '0114326', '0105409', '0105316', '0152366', '0157908']

def entropy(s):
	p, lns = Counter(s), float(len(s))
	return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def estimate_component_probs(activation_map, vox_prob_maps):
	"""
	Estimates component probabilities for a brain volume.
	Uses Expecation Maximization to calculate pr(component|volume) based
	on values in activation_map and vox_prob_maps.
	Parameters
	----------
	actiation_map : array_like
		A 2D array (n_voxels, 1) of voxel intensities.
	vox_prob_maps : array_like
		A 2D array (n_voxels, k) of k pr(voxel|component) maps.
	Returns
	-------
	array_like
		An array (of length k) of pr(component|volume) estimates.
	Notes
	-----
	Prior is that pr(component|volume) is the same for all components.
	"""
	num_components = vox_prob_maps.shape[-1]
	current_estimate = np.zeros((1, num_components)) + np.true_divide(1, num_components)

	for i in xrange(1000):
		print i
		q = vox_prob_maps * current_estimate
		q_sum = np.sum(q, axis=-1) + np.spacing(1) 
		q_frac = np.true_divide(1, q_sum)
		q_frac = np.reshape(q_frac, (q_frac.shape[0],1))
		q_prod = q * q_frac 
		
		expected_count_components = np.sum(np.multiply(q_prod, activation_map), axis=0)
		pr_components = np.true_divide(expected_count_components, np.sum(expected_count_components))
		change = np.max(np.absolute(current_estimate - pr_components))
		current_estimate = np.reshape(pr_components, (1, num_components))

		if change < 1e-3:
			#print "EM has converged as of iteration %s, with a maximum absolute difference of %s." % (i, change)
			break
	
	return current_estimate.ravel()

def estimate(brain_data, vox_prob_maps, scale_factor=10):
	
	"""
	Estimate pr(component|activation) using Expecation Maximization.
	
	Before estimation, brain_data is scaled, discretized, and negative values are set to 0.
	
	Parameters
	----------
	brain_data : array_like
		A single volume activation/intensity map with shape (1, n_voxels).
	vox_prob_maps : array_like
		A stack of K pr(voxel|component) maps with shape (K, n_voxels).
	
	scale_factor : int, optional
		A constant by which to scale values in brain_data. Default is 10.
		
	Returns
	-------
	array_like
		An array (of length K) of pr(component|volume) estimates.
	"""
	brain_data = np.round(brain_data * scale_factor)
	brain_data[brain_data < 0] = 0
	estimates = []

	if brain_data.shape[-1] > 1:
		print("Brain data is 4D; iterating over volumes...")
		n_vox = brain_data.shape[0]
		for i_vol in xrange(brain_data.shape[-1]):
			print("Reshaping data for volume %s..." % str(i_vol))
			activation_map = np.reshape(brain_data[...,i_vol], (n_vox, 1))
			print("Calculating estimates for volume %s..." % str(i_vol))
			estimates.append(estimate_component_probs(activation_map, vox_prob_maps))
	else:
		activation_map = brain_data
		print("Calculating estimates...")
		estimates.append(estimate_component_probs(activation_map, vox_prob_maps))
	return estimates

def get_2d_volume_data(nifti_file):
	"""
	Load image data from a NIfTI file and reshape the data array to n_vox by n_vol.
	
	Parameters
	----------
	nifti_file : str
		Full path to a NIfTI file.
		
	Returns
	-------
	array_like
		Image data as an array with shape n_vox by n_vol.
	
	"""
	temp_data = nib.load(nifti_file).get_data().astype('float64')
	if len(temp_data.shape) == 3:
		return np.reshape(temp_data, (np.prod(temp_data.shape), 1))
	elif len(temp_data.shape) == 4:
		return np.reshape(temp_data, (np.prod(temp_data.shape[:3]), temp_data.shape[-1]))
	else:
		print("Error: Image file must be 3D (x, y, z) or 4D (x, y, z, t).")

def run_component_estimation(subject,num_comps=12,ignore_flex=False):
	global subject_dir
	try:
		e = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_%s_%s_engagement.npy'%(subject,num_comps,ignore_flex))
		e = e.reshape(-1,num_comps)
	except:
		component_file = get_2d_volume_data('/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Yeo_%sComp_PrActGivenComp_FSL_MNI152_2mm.nii.gz' %(num_comps))
		epi_data = brain_graphs.load_subject_time_series(subject_dir.replace('SUBJECT',str(subject)))
		if ignore_flex != False:
			flex = '/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Flexibility/YeoMD_%scomp_FSL_MNI152_thresh1e-5.nii' %(num_comps)
			flex = nib.load(flex).get_data().astype('float64')
			epi_data[flex>=ignore_flex,:] = 0.
		e = []
		for i in range(epi_data.shape[3]):
			brain_data = epi_data[:,:,:,i]
			brain_data = np.reshape(brain_data, (np.prod(brain_data.shape),1))
			engagement = estimate(brain_data,component_file)
			e.append(engagement)
			del brain_data
			del engagement
		np.save('/home/despoB/mb3152/dynamic_mod/component_activation/%s_%s_%s_engagement.npy'%(subject,num_comps,ignore_flex), np.array(e))
	return e

def analyze_estimates(subject,project,num_comps,ignore_flex):
	e = run_component_estimation(subject,num_comps,ignore_flex)
	e_stats = np.array(np.zeros([3,num_comps]))
	for i in range(num_comps):
		data = e[:,i]
		e_stats[0,i] = np.max(data)
		e_stats[1,i] = np.mean(data)
		e_stats[2,i] = len(data[data>1./num_comps])
	estimates = []
	for i in range(e.shape[0]):
		estimates.append(len(e[i][e[i]>=1./(num_comps+1)]))
	return e_stats

def array_to_2mm_mni(array,filename):
	image = nib.load('/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz')
	data = image.get_data()
	data[:,:,:,] = array[:,:,:,]
	nib.save(image,filename)

def gordon_communities():
	df = pd.read_excel('Parcels.xlsx')
	for i,com in enumerate(np.unique(df.Community.values)):
		name_dict[com] = i
	Community_Number = np.zeros((333))
	for i in range(333):
		Community_Number[i] = name_dict[df.Community[i]]
	return Community_Number

def flex_activity(subject, num_comps = 12,ignore_flex=4):
	flex = '/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Flexibility/YeoMD_%scomp_FSL_MNI152_thresh1e-5.nii' %(num_comps)
	flex = nib.load(flex).get_data().astype('float64')
	components_engaged_var = []
	components_engaged_mean = []
	flex_activity = []
	non_flex_activity= []
	component_engagement = run_component_estimation(subject=subject,num_comps=num_comps,ignore_flex=ignore_flex)
	epi_data = brain_graphs.load_subject_time_series(subject_dir.replace('SUBJECT',str(subject)))
	epi_data[np.max(epi_data,axis=3)==0.0] = np.nan
	for i in range(epi_data.shape[-1]):
		brain_data = epi_data[:,:,:,i]
		non_flex_activity.append(np.nanmean(brain_data[flex<ignore_flex]))
		flex_activity.append(np.nanmean(brain_data[flex>=ignore_flex]))
		engagement = np.array(component_engagement[i])
		components_engaged_var.append(1-np.std(engagement))
		components_engaged_mean.append(len(engagement[engagement>(1./float(num_comps))]))
	print pearsonr(components_engaged_var,flex_activity)
	print pearsonr(components_engaged_var,non_flex_activity)
	x = components_engaged_var
	y = components_engaged_mean
	z = flex_activity
	w = non_flex_activity
	np.save('/home/despoB/mb3152/dynamic_mod/component_activation/%s_flex_data_%s_%s.npy'%(subject,num_comps,ignore_flex), np.array([x,y,z,w]))

def pc_activity(subject,num_comps = 12,ignore_flex=4):
	parcel_path = '/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %('power')
	epi_data = brain_graphs.load_subject_time_series(subject_dir.replace('SUBJECT',str(subject)))
	data = pd.read_csv('/home/despoB/mb3152/modularity/mmc3.csv')
	data['new'] = np.ones(len(data))
	for x2,y2,z2,pc in zip(data.X2,data.Y2,data.Z2,data.PC):
		data.new[data[data.X1==x2][data.Z1==z2][data.Y1==y2].index[0]] = pc
	values_dict = data.new.to_dict()

	# f = open('/home/despoB/mb3152/dynamic_mod/matrices/%s_graph.pkl'%(atlas))
	# g_partition = pickle.load(f)[0]
	# f.close()
	# matrix = brain_graphs.time_series_to_matrix(subject_time_series=epi_data,voxel=False,parcel_path=parcel_path)
	# graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost=.1)
	# partition = brain_graphs.brain_graph(VertexClustering(graph, membership=g_partition.community.membership))
	# partition = brain_graphs.recursive_network_partition(matrix=brain_graphs.time_series_to_matrix(epi_data,parcel_path),parcel_path=parcel_path,max_cost=.5)[0]
	template = np.array(nib.load(parcel_path).get_data())
	components_engaged_var = []
	components_engaged_mean = []
	high_pc_activity = []
	low_pc_activity = []
	epi_data[np.std(epi_data,axis=3)==0.] = np.nan
	pc_array = nib.load('/home/despoB/mb3152/modularity/figures/pc.nii').get_data()
	component_engagement = run_component_estimation(subject,num_comps,ignore_flex=ignore_flex)
	pc_array = np.zeros(pc_array.shape)
	pc_array[:,:,:,] = np.nan
	# wmd_array = pc_array.copy()
	for i in range(len(values_dict.values())):
		pc_array[template==i+1] = values_dict.values()[i]
	# for i in range(len(partition.wmd)):
	# 	wmd_array[template==i+1] = partition.wmd[i]
	# pc_array[np.nanstd(epi_data,axis=3)==0] = np.nan
	# pc_thresh = np.percentile(pc_array[np.isnan(pc_array)==False],80,interpolation='lower')
	# wmd_thresh = 1e-5
	# connector_array = pc_array.copy()
	# connector_array[pc_array<pc_thresh] = np.nan
	# connector_array[wmd_array<wmd_thresh] = np.nan
	# connector_ay[np.isnan(connector_array)==False] = 1
	pc_thresh = 4
	# wmd_thresh = 1e-5
	print pc_thresh
	for i in range(epi_data.shape[-1]):
		brain_data = epi_data[:,:,:,i]
		high_pc_activity.append(np.nanmean(brain_data[pc_array>=pc_thresh]))
		low_pc_activity.append(np.nanmean(brain_data[pc_array<pc_thresh]))
		engagement = component_engagement[i]
		components_engaged_var.append(1-np.std(engagement))
		components_engaged_mean.append(len(engagement[engagement>(1./float(num_comps))]))
	print pearsonr(components_engaged_var,high_pc_activity)
	print pearsonr(components_engaged_var,low_pc_activity)
	x = components_engaged_var
	y = components_engaged_mean
	z = high_pc_activity
	w = low_pc_activity
	pc_thresh=  'power' + str(pc_thresh)
	np.save('/home/despoB/mb3152/dynamic_mod/component_activation/%s_pc_data_%s_%s_%s.npy'%(subject,num_comps,ignore_flex,pc_thresh), np.array([x,y,z,w]))

def read_results(subjects=None,atlas='Shen',a_type='flex',corr_type='var',num_comps=12,ignore_flex=False,pc_thresh=8000,plot=False):
	if subjects == None:
		subjects = []
		if a_type == 'flex':
			subject_files = glob.glob('/home/despoB/mb3152/dynamic_mod/component_activation/**_%s_data_%s_%s.npy' %(a_type,num_comps,ignore_flex))
			for subject_file in subject_files:
				subjects.append(subject_file.split('_')[2].split('/')[1])  		
		else:
			subject_files = glob.glob('/home/despoB/mb3152/dynamic_mod/component_activation/0**_%s_data_%s_%s_%s.npy' %(a_type,num_comps,ignore_flex,pc_thresh))
			for subject_file in subject_files:
				subjects.append(subject_file.split('_')[2].split('/')[1])  	   
	d = []
	for i in subject_files:
		d.append(list(np.load(i)))
	# for subject in subjects:
	# 	if a_type != 'flex':
	# 		d = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_%s_data_%s_%s.npy' %(subject,a_type,num_comps,ignore_flex))    
	# 	else:
	# 		d = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_%s_data_%s_%s.npy' %(subject,a_type,num_comps,ignore_flex))  

	# 	data.append(d)
	data = np.array(d)
	print 'High Flex'
	for subject in range(len(subjects)):
		if corr_type == 'var':
			print pearsonr(data[subject,0,:],data[subject,2,:]),subjects[subject]
		else:
			print pearsonr(data[subject,1,:],data[subject,2,:]),subjects[subject]
	print '______________________'
	print 'Low Flex'
	for subject in range(len(subjects)):
		if corr_type == 'var':
			print pearsonr(data[subject,0,:],data[subject,3,:]),subjects[subject]
		else:
			print pearsonr(data[subject,1,:],data[subject,3,:]),subjects[subject]
	print '______________________'
	print 'Difference'
	for subject in range(len(subjects)):
		if corr_type == 'var':
			print pearsonr(data[subject,0,:],data[subject,2,:])[0] - pearsonr(data[subject,0,:],data[subject,3,:])[0], subjects[subject]
		else:
			print pearsonr(data[subject,1,:],data[subject,2,:])[0] - pearsonr(data[subject,1,:],data[subject,3,:])[0], subjects[subject]
	print '______________________'
	if corr_type == 'var':
		print pearsonr(data[:,0,:].reshape(-1),data[:,2,:].reshape(-1))
		print pearsonr(data[:,0,:].reshape(-1),data[:,3,:].reshape(-1))
	else:
		print pearsonr(data[:,1,:].reshape(-1),data[:,2,:].reshape(-1))
		print pearsonr(data[:,1,:].reshape(-1),data[:,3,:].reshape(-1))
	if plot == True:
		df = pd.DataFrame(data=np.array([data[:,0,:].reshape(-1),data[:,2,:].reshape(-1),data[:,3,:].reshape(-1)]).transpose(),columns=['Components Engaged','High Flexibility','Low Flexibility'])
		g = sns.regplot('Components Engaged','High Flexibility', df,color='Red',scatter=True,scatter_kws={'alpha':.05})
		# plt.xlim(min(df['Components Engaged']),max(df['Components Engaged']))
		# plt.yticks([])
		plt.xticks([])
		# plt.ylim(min(df['High Flexibility']),max(df['High Flexibility']))
		plt.ylim(-100,100)
		plt.ylabel('Mean Activity at Connector Areas')
		plt.xlabel("Cognitive Components Engaged")
		sns.despine()
		plt.show()
		g = sns.regplot('Components Engaged','Low Flexibility', df,color='Blue',scatter_kws={'alpha':.05})
		# plt.xlim(min(df['Components Engaged']),max(df['Components Engaged']))
		# print min(df['Components Engaged'])
		# print max(df['Components Engaged'])
		# plt.ylim(min(df['Low Flexibility']),max(df['Low Flexibility']))
		plt.ylim(-100,100)
		# plt.yticks([])
		plt.xticks([])
		plt.ylabel('Mean Activity at Local Areas')
		plt.xlabel("Cognitive Components Engaged")
		sns.despine()
		plt.show()

def make_partition(subjects,atlas,path):
	matrix = []
	pc = []
	for subject in subjects:
		matrix.append(np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_matrix.npy'%(subject,atlas)))
	matrix = np.nanmean(matrix,axis=0)
	# partition_graph = brain_graphs.recursive_network_partition(parcel_path=None,subject_paths=[],matrix=matrix.copy(),graph_cost=.1,max_cost=.3,min_cost=0.02,min_community_size=5,min_weight=1.)
	partition_graph = brain_graphs.partition_avg_costs(matrix,costs=np.array(range(10,101))/1000.,min_community_size=5,graph_cost=.1)
	if path != None:
		brain_graphs.save_graph(path,partition_graph)
	return partition_graph

def draw_adjacency_matrix(matrix, membership, names):
	swap_dict = {}
	index = 0
	corr_mat = np.zeros((matrix.shape))
	new_names = []
	x_ticks = []
	y_ticks = []
	for i in np.unique(membership):
		for node in np.where(membership==i)[0]:
			swap_dict[node] = index
			index = index + 1
			new_names.append(names[node])
	y_names = []
	x_names = []
	old_name = 0
	for i,name in enumerate(new_names):
		if name == old_name:
			continue
		old_name = name
		y_ticks.append(i)
		x_ticks.append(len(new_names)-i)
		y_names.append(name)
		x_names.append(name)
	for i in range(len(swap_dict)):
		for j in range(len(swap_dict)):
			corr_mat[swap_dict[i],swap_dict[j]] = matrix[i,j]
			corr_mat[swap_dict[j],swap_dict[i]] = matrix[j,i]
	membership.sort()
	sns.set(context="paper", font="monospace")
	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(12, 9))
	# Draw the heatmap using seaborn
	y_names.reverse()
	sns.heatmap(corr_mat,vmin=-.25,vmax=.25,square=True,yticklabels=y_names,xticklabels=x_names,linewidths=0.0,)
	ax.set_yticks(x_ticks)
	ax.set_xticks(y_ticks)
	# Use matplotlib directly to emphasize known networks
	networks = membership
	for i, network in enumerate(networks):
		if network != networks[i - 1]:
			ax.axhline(len(networks) - i, c='black',linewidth=2)
			ax.axvline(i, c='black',linewidth=2)
	f.tight_layout()
	plt.show()
	plt.close()

def dynamic_graph_analyses(subjects=None,atlas='shen',msc_cost=0.1,hub_cost = .1,window_size=100,make_images=False):
	atlas = 'power'
	if atlas == 'power':
		known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
		known_membership[known_membership==-1] = 0
		colors = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[34].values)
		names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)
	num_nodes = len(known_membership)
	gamma = 1.0
	omega = .1
	msc_cost = 0.1
	window_size = 100
	if subjects == None:
		subjects = []
		subject_paths = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/*_%s_%s_%s_msc_%s_%s.npy' %(atlas,window_size,msc_cost,gamma,omega))
		for s in subject_paths:
			subjects.append(s.split('/')[-1].split('_')[0])
	subject_mods = [] #individual subject modularity values
	subject_changes = [] #communities at each node
	subject_num_changes = [] #number of changes at a node
	subject_pcs = [] #subjects PCs
	subject_bms = [] #between module strength
	subject_wms = [] #within module strength
	matrices = []
	for subject in subjects:
		print subject
		msc = np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_%s_msc_%s_%s.npy' %(subject,atlas,window_size,msc_cost,gamma,omega))
		s_matrix = np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_matrix.npy'%(subject,atlas))
		matrices.append(s_matrix.copy())
		s_matrix[np.isnan(s_matrix)] = 0.0
		s_mods = []
		s_pcs = []
		s_bms = []
		s_wms = []
		cost = .25
		while True:
			graph = brain_graphs.matrix_to_igraph(s_matrix.copy(),cost)
			graph = graph.community_infomap(edge_weights='weight')
			graph = brain_graphs.brain_graph(graph)
			s_mods.append(graph.community.modularity)
			s_pcs.append(np.array(graph.pc))
			edges = graph.community.graph.get_edgelist()
			for edge in edges:
				if graph.community.membership[edge[0]] == graph.community.membership[edge[1]]:
					s_wms.append(s_matrix[edge[0],edge[1]])
				else:
					s_bms.append(s_matrix[edge[0],edge[1]])
			if cost < .1:
				break
			cost = cost - 0.01
		subject_mods.append(np.mean(s_mods))
		subject_pcs.append(np.nanmean(s_pcs,axis=0))
		subject_bms.append(np.mean(s_bms))
		subject_wms.append(np.mean(s_wms))
		num_communities = []
		for i in range(msc.shape[1]):
			num_communities.append(len(np.unique(msc[:,i])))
		subject_changes.append(np.array(num_communities))
		num_changes = []
		for i in range(msc.shape[1]):
			c = 0
			for t in range(msc.shape[0]):
				if t == 0:
					continue
				if msc[t,i] == msc[t-1,i]:
					continue
				c = c + 1
			num_changes.append(c)
		subject_num_changes.append(np.array(num_changes))
	subject_changes = np.array(subject_changes)
	subject_num_changes = np.array(subject_num_changes)
	subject_mods = np.array(subject_mods)
	subject_pcs = np.array(subject_pcs)
	subject_pcs[np.isnan(subject_pcs)] = 0.0
	matrices = np.array(matrices)
	# correlate PC with the strength of different edges
	pc_edge_corr = np.zeros((subject_pcs.shape[1],subject_pcs.shape[1],subject_pcs.shape[1]))
	mean_subject_pcs = np.nanmean(subject_pcs,axis=0)
	for i in range(subject_pcs.shape[1]):
		for n1,n2 in combinations(range(subject_pcs.shape[1]),2):
			val = pearsonr(subject_pcs[:,i],matrices[:,n1,n2])[0]
			pc_edge_corr[i,n1,n2] = val
			pc_edge_corr[i,n2,n1] = val
	pc_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),75)
	connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)>pc_thresh)[0]
	low_pc_edge_matrix = np.nanmean(pc_edge_corr[np.where(mean_subject_pcs<pc_thresh)],axis=0)
	high_pc_edge_matrix = np.nanmean(pc_edge_corr[np.where(mean_subject_pcs>=pc_thresh)],axis=0)
	# low_pc_edge_matrix = np.mean(pc_edge_corr[np.where(mean_subject_pcs<.2)],axis=0)
	# high_pc_edge_matrix = np.mean(pc_edge_corr[np.where(mean_subject_pcs>=.45)],axis=0)
	pc_edge_matrix = np.nanmean(pc_edge_corr,axis=0)
	pc_edge_matrix[np.isnan(pc_edge_matrix)] = 0.0
	high_pc_edge_matrix[np.isnan(high_pc_edge_matrix)]=0.0
	low_pc_edge_matrix[np.isnan(low_pc_edge_matrix)]=0.0
	1/0
	
	#subjcet by node by community
	community_mod = np.zeros((len(subjects),len(np.unique(known_membership))))
	#for each subject
	pc_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),75)
	connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)>pc_thresh)[0]
	mean_connector_pc = np.zeros(len(subjects))
	mean_non_connector_pc = np.zeros(len(subjects))
	for i,subject in enumerate(subjects):
		connector = []
		non_connector = []
		for node in range(num_nodes):
			if node in connector_nodes:
				connector.append(subject_pcs[i,node])
			else:
				non_connector.append(subject_pcs[i,node])
		mean_connector_pc[i] = np.nanmean(connector)
		mean_non_connector_pc[i] = np.nanmean(non_connector)
		#for each community
		for community in np.unique(known_membership):
			all_nodes = np.array(range(num_nodes))
			community_corr_mat = []
			community_nodes = np.where(known_membership==community)[0]
			non_community_nodes = np.where(known_membership!=community)[0]
			wcd = float(np.nanmean(matrices[i][np.ix_(community_nodes,community_nodes)]))
			bcd = float(np.nanmean(matrices[i][np.ix_(non_community_nodes,community_nodes)]))
			community_mod[i,community] = wcd/bcd
	community_mod[np.isnan(community_mod)] = np.nanmean(community_mod)
	#which communities modularity correlates with the PC of connector nodes






			#for each node we have
		# for node in range(subject_pcs.shape[1]):
		# 	#get the relationship between node's PC and modulairty of that network
		# 	#are certain networks' modularity impacted more or less by the PC of connector nodes or nodes in general?
		# 	pc_specific_community[node,community] = pearsonr(subject_pcs)
	#do specific nodes have specific relationships to modularity of each network?
	#which nodes changes correlate with modularity?
	mod_change_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_change_corr[i] = pearsonr(subject_mods,subject_changes[:,i])[0]
	#which nodes pc correlate with modularity?
	mod_pc_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_pc_corr[i] = pearsonr(subject_mods,subject_pcs[:,i])[0]
	# which nodes number of changes correlate with modularity
	mod_num_change_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_num_change_corr[i] = pearsonr(subject_mods,subject_num_changes[:,i])[0]

	print 'Modularity X Mean WMS: ' + str(pearsonr(subject_mods,subject_wms))
	print 'Modularity X Mean BMS: ' + str(pearsonr(subject_mods,subject_bms))
	print 'Modularity X WMS/BMS: ' + str(pearsonr(subject_mods,np.array(subject_wms)/np.array(subject_bms)))
	print 'Modularity X Mean PC: ' + str(pearsonr(subject_mods,np.nanmean(subject_pcs,axis=1)))

	# if make_images == True:
	#     df = pd.DataFrame(data=np.array([subject_mods,np.nanmean(subject_pcs,axis=1)]).transpose(),columns=['Modularity','Mean Participation Coefficient'])
	#     sns.regplot('Modularity','Mean Participation Coefficient',df,scatter=True)
	#     plt.show()
	#     brain_graphs.make_image('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas),'/home/despoB/mb3152/dynamic_mod/brain_figures/mod_num_change_corr_%s' %(atlas),mod_num_change_corr*10000)
	#     brain_graphs.make_image('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas),'/home/despoB/mb3152/dynamic_mod/brain_figures/mod_change_corr_%s' %(atlas),mod_change_corr*10000)
	#     brain_graphs.make_image('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas),'/home/despoB/mb3152/dynamic_mod/brain_figures/mod_pc_corr_%s' %(atlas),mod_pc_corr*10000)
	#     brain_graphs.make_image('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas),'/home/despoB/mb3152/dynamic_mod/brain_figures/pc_%s' %(atlas),np.nanmean(subject_pcs,axis=0)*10000)

	#     subject_num_changes = np.array(subject_num_changes)
	#     mod_num_change_corr = np.zeros(subject_pcs.shape[1])
	#     for i in range(subject_pcs.shape[1]):
	#     	mod_num_change_corr[i] = pearsonr(subject_mods,subject_num_changes[:,i])[0]
	#     brain_graphs.make_image('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas),'/home/despoB/mb3152/dynamic_mod/brain_figures/mod_num_change_corr_%s' %(atlas),mod_num_change_corr*10000)

def time_point_graph_analyses(subject,atlas='power',num_comps=12,ignore_flex=False):
	def run(subject,parcel_path):
		engagement = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_%s_%s_%s_engagement.npy'%(project,subject,num_comps,ignore_flex))
		engagement = 1-np.array(np.std(engagement.reshape(engagement.shape[0],num_comps),axis=1))
		high_points = np.argwhere(engagement>=np.percentile(engagement,75))
		low_points = np.argwhere(engagement<=np.percentile(engagement,25))
		epi_data = brain_graphs.load_subject_time_series(subject_dir.replace('SUBJECT',str(subject)))
		partition = brain_graphs.recursive_network_partition(graph_cost = 0.05, matrix=brain_graphs.time_series_to_matrix(epi_data,parcel_path),parcel_path=parcel_path,max_cost=.5)[0]# make full graph from normal time series (maybe impose partition?) perhaps use mean of DC or show similar results with that and normal FC.
		membership = partition.community.membership
		graph = brain_graphs.matrix_to_igraph(brain_graphs.time_series_to_matrix(epi_data[:,:,:,high_points].reshape(91, 109, 91, 225),parcel_path),.1)
		high_partition = brain_graphs.brain_graph(VertexClustering(graph, membership=membership))
		graph = brain_graphs.matrix_to_igraph(brain_graphs.time_series_to_matrix(epi_data[:,:,:,low_points].reshape(91, 109, 91, 225),parcel_path),.1)
		low_partition = brain_graphs.brain_graph(VertexClustering(graph, membership=membership))
		return partition,low_partition,high_partition
	parcel_path= '/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas)
	p,l,h = run(subject,parcel_path)
	f = open('/home/despoB/mb3152/dynamic_mod/graphs/%s_%s_%s_graph.pkl'%(subject,atlas,ignore_flex),'w')
	pickle.dump([p,l,h],f)
	f.close()

def group_time_point_graph_analyses(subjects,atlas,ignore_flex):
	l_mod = []
	h_mod = []
	mod = []
	added_p_hub = []
	added_c_hub = []
	degree_diffs = []
	for subject in subjects:
		try:
			f = open('/home/despoB/mb3152/dynamic_mod/graphs/%s_%s_%s_graph.pkl'%(subject,atlas,ignore_flex))
			p,lp,hp = pickle.load(f)
			f.close()
		except:
			continue
		h_mod.append(hp.community.modularity)
		mod.append(p.community.modularity)
		l_mod.append(lp.community.modularity)

		degree_diff = np.array(hp.community.graph.degree()-np.array(p.community.graph.degree()))
		degree_diffs.append(degree_diff)
		pc_thresh = np.nanmean(p.pc)
		connector_hubs = np.intersect1d(np.argwhere(p.pc>pc_thresh),np.argwhere(p.wmd>0))
		sat_connectors = np.intersect1d(np.argwhere(p.pc>pc_thresh),np.argwhere(p.wmd<=0))
		provincial_hubs = np.intersect1d(np.argwhere(p.pc<=pc_thresh),np.argwhere(p.wmd>0))
		nodes = np.intersect1d(np.argwhere(p.pc<=pc_thresh),np.argwhere(p.wmd<=0))
		# print subject
		if np.isnan(np.mean(degree_diff[connector_hubs])) or np.isnan(np.mean(degree_diff[provincial_hubs])):
			continue
		added_c_hub.append(np.mean(degree_diff[connector_hubs]))
		added_p_hub.append(np.mean(degree_diff[provincial_hubs]))
		print ttest_ind(degree_diff[provincial_hubs],degree_diff[connector_hubs])

def nodes_to_components(atlas,num_nodes=12):
	template = nib.load('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas)).get_data()
	yeo_flat_comps = []
	yeo = nib.load('/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Yeo_%sComp_PrActGivenComp_FSL_MNI152_2mm.nii.gz' %(num_comps)).get_data()
	yeo_comps = []
	for node in np.unique(template):
		values = []
		indices = np.argwhere(template==node)
		for c in range(num_comps):
			comp_values = []
			for i in indices:
				comp_values.append(yeo[i[0],i[1],i[2],c])
			values.append(np.sum(comp_values))
		yeo_comps.append(np.argmax(values))
	return yeo_comps

"""
Get Subjects and Paths
"""
# num_comps = 12
# ignore_flex = False
# atlas = 'shen'
# subjects = []
# subject_files = glob.glob('/home/despoB/mb3152/dynamic_mod/component_activation/**%s_%s_engagement.npy' %(num_comps,ignore_flex))
# for subject_file in subject_files:
#     subjects.append(subject_file.split('_')[2].split('/')[1])
# subject_paths = []
# for subject in subjects:
# 	subject_paths.append(subject_dir.replace('SUBJECT',str(subject)))

"""
Single Graph
"""
# atlas = 'shen'
# subject = str(sys.argv[1])
# parcel_path = '/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas)
# subject_path = subject_dir.replace('SUBJECT',subject)
# subject_time_series = brain_graphs.load_subject_time_series(subject_path)
# matrix = brain_graphs.time_series_to_matrix(subject_time_series,parcel_path,voxel=False,fisher=False,out_file=None)
# np.save('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_matrix.npy',matrix)

"""
Group Graph
"""
# parcel_path = '/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas)
# brain_object = brain_graphs.recursive_network_partition(parcel_path,subject_paths=subject_paths,matrix=None,graph_cost=.1,max_cost=.1,min_cost=0.01,min_community_size=5)
# f = open('/home/despoB/mb3152/dynamic_mod/matrices/%s_graph.pkl'%(atlas),'w')
# pickle.dump(brain_object,f)
# f.close()

"""
Dynamic Graph Analyses
"""

# dynamic_graph_analyses(subjects=[sys.argv[1]],atlas='power',num_comps=12,ignore_flex=False,window_size=35)

"""
Component Engagement
"""
# e_stats = np.zeros(shape=(12,len(subjects)*900))
# for i in range(12):
# 	temp = []
# 	for subject in subjects:
# 		e = run_component_estimation(subject,12,False).transpose()
# 		temp.extend(e[i])
# 	e_stats[i,:] = temp
# for ix in range(12):
# 	print ix + 1
# 	for ij in range(12):
# 		if ij == ix:
# 			continue
# 		print ij+1, pearsonr(e_stats[ix],e_stats[ij])[0]

"""
Run PRICE
"""
# run_component_estimation(sys.argv[1],ignore_flex=False)
# run_component_estimation(sys.argv[1],ignore_flex=3.25)
# flex_activity(sys.argv[1],ignore_flex=False)
# flex_activity(sys.argv[1],ignore_flex=4)
# pc_activity(sys.argv[1],ignore_flex=False,atlas='power')
# pc_activity(sys.argv[1],ignore_flex=4,atlas='power')
 
"""
multi-slice
"""
# subject = str(sys.argv[1])
# atlas = 'power'
# gamma = 1.0
# omega = .1
# cost = 0.1
# window_size = 100
# parcel_path = '/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas)
# subject_path = subject_dir.replace('SUBJECT',subject)
# window_size = 100
# subjects = []
# subject_paths = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/*%s_%s_%s_msc_%s_%s.npy*' %(atlas,window_size,cost,gamma,omega))
# for s in subject_paths:
# 	subjects.append(s.split('/')[-1].split('_')[0])
# if subject not in subjects:
# 	subject_time_series = brain_graphs.load_subject_time_series(subject_path)
# 	matrix = brain_graphs.time_series_to_matrix(subject_time_series,parcel_path,voxel=False,fisher=False,out_file=None)
# 	np.save('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_matrix.npy'%(subject,atlas),matrix)
# 	matrix = brain_graphs.time_series_to_ewmf_matrix(subject_time_series=subject_time_series,parcel_path=parcel_path,window_size=window_size,out_file=None)
# 	del subject_time_series
# 	out_file = '/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_%s_msc_%s_%s.npy' %(subject,atlas,window_size,cost,gamma,omega)
# 	brain_graphs.multi_slice_community(matrix,cost,out_file)


"""
combine brain figures
"""

# files = ['mod_change_corr_**.nii','mod_pc_corr_**.nii','pc_**.nii','mod_num_change_corr_**.nii']
# names = ['mod_change_corr_avg','mod_pc_corr_avg','pc_avg','mod_num_change_corr']
# # files = ['mod_change_corr_**.nii','mod_pc_corr_**.nii','pc_**.nii']
# # names = ['mod_change_corr_avg','mod_pc_corr_avg','pc_avg']
# for f,name in zip(files,names):
# 	images = glob.glob('/home/despoB/mb3152/dynamic_mod/brain_figures/' + f)
# 	data = []
# 	for image in images:
# 		print image.split('_')[:-1]
# 		image = nib.load(image)
# 		data.append(image.get_data())
# 	new_img = nib.Nifti1Image(np.nanmean(data,axis=0), image.affine, image.header)
# 	nib.save(new_img, "/home/despoB/mb3152/dynamic_mod/brain_figures/%s" %(name))

# blurs = [18,20,22,24]
# for name in names:
# 	for blur in blurs:
# 		os.system('3dBlurToFWHM -input /home/despoB/mb3152/dynamic_mod/brain_figures/%s.nii -FWHM %s -prefix /home/despoB/mb3152/dynamic_mod/brain_figures/%s_%s.nii -overwrite' %(name,blur,name,blur))
# 1/0
subjects = None
atlas = 'gordon'
if atlas == 'power':
	known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
	known_membership[known_membership==-1] = 0
	colors = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[34].values)
	names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)
	
if atlas =='gordon':
	df = pd.read_excel('/home/despoB/mb3152/dynamic_mod/Parcels.xlsx')
	names = df.Community
	known_membership = np.zeros(len(names))
	for i,c in enumerate(np.unique(names)):
		known_membership[df.ParcelID[df.Community==c]-1]=i
	names[names=='None'] = 'Uncertain'
 
num_nodes = len(known_membership)
gamma = 1.0
omega = .1
msc_cost = 0.1
window_size = 100
if subjects == None:
	subjects = []
	subject_paths = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/*_%s_%s_%s_msc_%s_%s.npy' %(atlas,window_size,msc_cost,gamma,omega))
	for s in subject_paths:
		subjects.append(s.split('/')[-1].split('_')[0])
subject_mods = [] #individual subject modularity values
subject_changes = [] #communities at each node
subject_num_changes = [] #number of changes at a node
subject_pcs = [] #subjects PCs
subject_bms = [] #between module strength
subject_wms = [] #within module strength
matrices = []
thresh_matrices = []
#run graph analyses 
for subject in subjects:
	print subject
	msc = np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_%s_msc_%s_%s.npy' %(subject,atlas,window_size,msc_cost,gamma,omega))
	s_matrix = np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_matrix.npy'%(subject,atlas))
	s_matrix[np.isnan(s_matrix)] = 0.0
	np.fill_diagonal(s_matrix,0.0)
	#z-score the matrices so that we can run correllations later. 
	thresh_matrices.append(scipy.stats.zscore(s_matrix.reshape(-1)).reshape((num_nodes,num_nodes)).copy())
	matrices.append(s_matrix.copy())
	s_mods = []
	s_pcs = []
	s_bms = []
	s_wms = []
	cost = .25
	while True:
		temp_matrix = s_matrix.copy()
		#make sure we normalize the weights so that they are equal across subjects
		graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,check_tri=True,interpolation='midpoint',normalize=True)
		graph = graph.community_infomap(edge_weights='weight')
		graph = brain_graphs.brain_graph(graph)
		s_mods.append(graph.community.modularity)
		s_pcs.append(np.array(graph.pc))
		edges = graph.community.graph.get_edgelist()
		for edge in edges:
			if graph.community.membership[edge[0]] == graph.community.membership[edge[1]]:
				s_wms.append(s_matrix[edge[0],edge[1]])
			else:
				s_bms.append(s_matrix[edge[0],edge[1]])
		if cost < .05:
			break
		cost = cost - 0.01
	subject_mods.append(np.mean(s_mods))
	subject_pcs.append(np.nanmean(s_pcs,axis=0))
	subject_bms.append(np.mean(s_bms))
	subject_wms.append(np.mean(s_wms))
	num_communities = []
	for i in range(msc.shape[1]):
		num_communities.append(len(np.unique(msc[:,i])))
	subject_changes.append(np.array(num_communities))
	num_changes = []
	for i in range(msc.shape[1]):
		c = 0
		for t in range(msc.shape[0]):
			if t == 0:
				continue
			if msc[t,i] == msc[t-1,i]:
				continue
			c = c + 1
		num_changes.append(c)
	subject_num_changes.append(np.array(num_changes))
subject_changes = np.array(subject_changes)
subject_num_changes = np.array(subject_num_changes)
subject_mods = np.array(subject_mods)
subject_pcs = np.array(subject_pcs)
subject_pcs[np.isnan(subject_pcs)] = 0.0
matrices = np.array(matrices)
thresh_matrices = np.array(thresh_matrices)

#make a matrix of each nodes PC correlation to all edges in the graph.

pc_edge_corr = np.zeros((subject_pcs.shape[1],subject_pcs.shape[1],subject_pcs.shape[1]))
mean_subject_pcs = np.nanmean(subject_pcs,axis=0)
for i in range(subject_pcs.shape[1]):
	for n1,n2 in combinations(range(subject_pcs.shape[1]),2):
		val = pearsonr(subject_pcs[:,i],thresh_matrices[:,n1,n2])[0]
		pc_edge_corr[i,n1,n2] = val
		pc_edge_corr[i,n2,n1] = val

pc_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),66)
connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)>=pc_thresh)[0]
non_connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)<pc_thresh)[0]
low_pc_edge_matrix = np.nanmean(pc_edge_corr[non_connector_nodes],axis=0)
high_pc_edge_matrix = np.nanmean(pc_edge_corr[connector_nodes],axis=0)
matrix = np.nansum([scipy.tril(low_pc_edge_matrix),scipy.triu(high_pc_edge_matrix)],axis=0)

#plot above
membership = known_membership.copy()
swap_dict = {}
index = 0
corr_mat = np.zeros((matrix.shape))
new_names = []
x_ticks = []
y_ticks = []
for i in np.unique(membership):
	for node in np.where(membership==i)[0]:
		swap_dict[node] = index
		index = index + 1
		new_names.append(names[node])
y_names = []
x_names = []
old_name = 0
for i,name in enumerate(new_names):
	if name == old_name:
		continue
	old_name = name
	y_ticks.append(i)
	x_ticks.append(len(new_names)-i)
	y_names.append(name)
	x_names.append(name)
for i in range(len(swap_dict)):
	for j in range(len(swap_dict)):
		corr_mat[swap_dict[i],swap_dict[j]] = matrix[i,j]
		corr_mat[swap_dict[j],swap_dict[i]] = matrix[j,i]
membership.sort()
sns.set(context="paper", font="monospace")
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))
# Draw the heatmap using seaborn
y_names.reverse()
sns.heatmap(corr_mat,vmin=-.25,vmax=.25,square=True,yticklabels=y_names,xticklabels=x_names,linewidths=0.0,)
ax.set_yticks(x_ticks)
ax.set_xticks(y_ticks)
# Use matplotlib directly to emphasize known networks
networks = membership
for i, network in enumerate(networks):
	if network != networks[i - 1]:
		ax.axhline(len(networks) - i, c='black',linewidth=2)
		ax.axvline(i, c='black',linewidth=2)
f.tight_layout()
plt.show()

1/0

"""
modulation analyses
"""

#Within and between network edge PC modulation weights in matrix for each network, for each node.
community_mod_high_wcd = np.zeros((len(connector_nodes),len(np.unique(known_membership))))
community_mod_low_wcd = np.zeros((len(non_connector_nodes),len(np.unique(known_membership))))
community_mod_high_bcd = np.zeros((len(connector_nodes),len(np.unique(known_membership))))
community_mod_low_bcd = np.zeros((len(non_connector_nodes),len(np.unique(known_membership))))
community_mod_bcd = np.zeros((num_nodes,len(np.unique(known_membership))))
community_mod_wcd = np.zeros((num_nodes,len(np.unique(known_membership))))
for i,n in enumerate(connector_nodes):
	for community in np.unique(known_membership):
		community_nodes = np.where(known_membership==community)[0]
		non_community_nodes = np.where(known_membership!=community)[0]
		wcd = float(np.nanmean(pc_edge_corr[n][np.ix_(community_nodes,community_nodes)]))
		bcd = float(np.nanmean(pc_edge_corr[n][np.ix_(non_community_nodes,community_nodes)]))
		community_mod_high_wcd[i][community] = wcd
		community_mod_high_bcd[i][community] = bcd
for i,n in enumerate(non_connector_nodes):
	for community in np.unique(known_membership):
		community_nodes = np.where(known_membership==community)[0]
		non_community_nodes = np.where(known_membership!=community)[0]
		wcd = float(np.nanmean(pc_edge_corr[n][np.ix_(community_nodes,community_nodes)]))
		bcd = float(np.nanmean(pc_edge_corr[n][np.ix_(non_community_nodes,community_nodes)]))
		community_mod_low_wcd[i][community] = wcd
		community_mod_low_bcd[i][community] = bcd
for i,n in enumerate(range(num_nodes)):
	for community in np.unique(known_membership):
		community_nodes = np.where(known_membership==community)[0]
		non_community_nodes = np.where(known_membership!=community)[0]
		wcd = float(np.nanmean(pc_edge_corr[n][np.ix_(community_nodes,community_nodes)]))
		bcd = float(np.nanmean(pc_edge_corr[n][np.ix_(non_community_nodes,community_nodes)]))
		community_mod_wcd[i][community] = wcd
		community_mod_bcd[i][community] = bcd

#PC is correlated positively with wcd and negatively with bcd
scipy.stats.ttest_ind(community_mod_wcd.reshape(-1),community_mod_bcd.reshape(-1))
#we can look at this relationship for connector nodes or non-connector nodes. Stronger at connector nodes for wcd, stronger at non for bcd
#between community degree modulation is a decrease for connector nodes, increase for non_connector nodes.
scipy.stats.ttest_ind(community_mod_high_wcd.reshape(-1),community_mod_low_wcd.reshape(-1))
#within community degree modulation is an increase for connector nodes, decrease for non_connector nodes.
scipy.stats.ttest_ind(community_mod_high_bcd.reshape(-1),community_mod_low_bcd.reshape(-1))

#plot violin if the values each communities modulation within and between.
#make this for low_pc_edge_matrix and high_pc_edge_matrix
violin_pc_edge_corr = []
for i in range(len(np.unique(known_membership))):
	violin_pc_edge_corr.append([])
for n in range(num_nodes):
	if n not in connector_nodes:
		continue
	for community in np.unique(known_membership.astype(int)):
		community_nodes = np.where(known_membership==community)[0]
		non_community_nodes = np.where(known_membership!=community)[0]
		wcd = pc_edge_corr[n][np.ix_(community_nodes,community_nodes)].reshape(-1)
		bcd = pc_edge_corr[n][np.ix_(community_nodes,non_community_nodes)].reshape(-1)
		# violin_pc_edge_corr[community].extend(wcd[wcd!=0])
		violin_pc_edge_corr[community].extend(bcd[bcd!=0])
		# violin_pc_edge_corr[community].append(np.nanmean(wcd[wcd!=0])/np.nanmean(bcd[bcd!=0]))
df = pd.DataFrame()
for i,name in enumerate(x_names):
	if i == 0:
		df = pd.DataFrame(violin_pc_edge_corr[i],columns=[x_names[i]])
		continue
	temp_df = pd.DataFrame(violin_pc_edge_corr[i],columns=[x_names[i]])
	df = pd.concat([df,temp_df])
med = df.median()
med.sort()
newdf = df[med.index]
newdf.boxplot()
plt.xticks(rotation=90)
plt.ylim((np.nanmin(df),np.nanmax(df)))
plt.tight_layout()
plt.yticks(size=16)
plt.xticks(size=16)
plt.show()

#Within and between network weights in real matrix for each network,
community_wcd = np.zeros((len(subjects),len(np.unique(known_membership))))
community_bcd = np.zeros((len(subjects),len(np.unique(known_membership))))
for i,subject in enumerate(subjects):
	for community in np.unique(known_membership):
		community_nodes = np.where(known_membership==community)[0]
		non_community_nodes = np.where(known_membership!=community)[0]
		wcd = float(np.nanmean(matrices[i][np.ix_(community_nodes,community_nodes)]))
		bcd = float(np.nanmean(matrices[i][np.ix_(non_community_nodes,community_nodes)]))
		community_wcd[i,community] = wcd
		community_bcd[i,community] = bcd
community_mod = community_wcd/community_bcd

#Correlation of each node's PC with bcd and wcd of each network, across subjects.
pc_by_individual_wcd = np.zeros((num_nodes,len(np.unique(known_membership))))
pc_by_individual_bcd = np.zeros((num_nodes,len(np.unique(known_membership))))
pc_by_individual_mod = np.zeros((num_nodes,len(np.unique(known_membership))))
for ix,i in enumerate(range(num_nodes)):
	for c,name in zip(range(len(np.unique(known_membership))),names):
		pc_by_individual_wcd[ix,c] = pearsonr(subject_pcs[:,i],community_wcd[:,c])[0]
		pc_by_individual_bcd[ix,c] = pearsonr(subject_pcs[:,i],community_bcd[:,c])[0]
		pc_by_individual_mod[ix,c] = pearsonr(subject_pcs[:,i],community_wcd[:,c]/community_bcd[:,c])[0]
df = pd.DataFrame(pc_by_individual_mod[connector_nodes],columns=x_names)
med = df.median()
med.sort()
newdf = df[med.index]
sns.violinplot(vals=newdf)
# plt.ylim((np.nanmin(df),np.nanmax(df)))
plt.tight_layout()
plt.xticks(rotation=90)
plt.yticks(size=16)
plt.xticks(size=16)
plt.show()

#correlate bcd of each set of networks with PC.
mod_pc_network_matrix = np.zeros((num_nodes,len(np.unique(known_membership)),len(np.unique(known_membership))))
for n in range(num_nodes):
	for c1,c2, in combinations(range(len(np.unique(known_membership))),2):
		community_nodes = np.where(known_membership==c1)[0]
		non_community_nodes = np.where(known_membership==c2)[0]
		bcds= []
		for s in range(thresh_matrices.shape[0]):
			bcds.append(np.nanmean(thresh_matrices[s][np.ix_(community_nodes,non_community_nodes)]))
		bcd = pearsonr(subject_pcs[:,n],bcds)[0]
		mod_pc_network_matrix[n,c1,c2] = bcd
		mod_pc_network_matrix[n,c2,c1] = bcd

low_pc_network_matrix = np.nanmean(mod_pc_network_matrix[non_connector_nodes],axis=0)
high_pc_network_matrix = np.nanmean(mod_pc_network_matrix[connector_nodes],axis=0)
matrix = np.nansum([scipy.tril(low_pc_network_matrix),scipy.triu(high_pc_network_matrix)],axis=0)

sns.heatmap(matrix,vmin=-.15,vmax=.15,square=True,yticklabels=x_names,xticklabels=x_names,linewidths=0.0,)
plt.tight_layout()
plt.show()


#does connector nodes PC correlate stronger with the wcd of each network?
scipy.stats.ttest_ind(pc_by_individual_wcd[connector_nodes],pc_by_individual_wcd[non_connector_nodes])
#does connector nodes PC correlate stronger with the bcd of each network?
scipy.stats.ttest_ind(pc_by_individual_bcd[connector_nodes],pc_by_individual_bcd[non_connector_nodes])
#Mean connectivity of each node to each network, relative to overall strength
community_stregth = np.zeros((len(subjects),num_nodes,len(np.unique(known_membership))))
for i,subject in enumerate(subjects):
	for n in range(num_nodes):
		for community in np.unique(known_membership):
			community_nodes = np.where(known_membership==community)[0]
			community_stregth[i,n,community] = np.nanmean(thresh_matrices[i,n,community_nodes])/np.nanmean(thresh_matrices[i,n,:])

# No relation beween connector node's PC modulation of single network with nodes' connectivity to that network, suggesting connector nodes work together. 
strength = np.nanmean(community_stregth,axis=0) #average connectivity to each module by each nodes across subjects
for c,n in zip(np.unique(known_membership),x_names):
	print n, scipy.stats.spearmanr(strength[connector_nodes,c].reshape(-1),pc_by_individual_bcd[connector_nodes,c].reshape(-1))
for c,n in zip(np.unique(known_membership),x_names):
	print n, scipy.stats.spearmanr(strength[connector_nodes,c].reshape(-1),pc_by_individual_wcd[connector_nodes,c].reshape(-1))
for c,n in zip(np.unique(known_membership),x_names):
	print n, scipy.stats.spearmanr(strength[connector_nodes,c].reshape(-1),pc_by_individual_mod[connector_nodes,c].reshape(-1))
for c,n in zip(np.unique(known_membership),x_names):
	print n, scipy.stats.spearmanr(strength[connector_nodes,c].reshape(-1),community_mod_high_wcd[:,c].reshape(-1))
for c,n in zip(np.unique(known_membership),x_names):
	print n, scipy.stats.spearmanr(strength[connector_nodes,c].reshape(-1),community_mod_high_bcd[:,c].reshape(-1))

#their modulation is not even more correlated with the networks they connect to than non-connector nodes.
connector_modulation_strength = []
non_connector_modulation_strength = []
for n in range(num_nodes):
	val = scipy.stats.spearmanr(strength[n],pc_by_individual_mod[n])[0]
	if n in connector_nodes:
		connector_modulation_strength.append(val)
	else:
		non_connector_modulation_strength.append(val)

#which specific networks are impacted by more by connector nodes PC values than non-connector nodes PC values?
result = scipy.stats.ttest_ind(pc_by_individual_wcd[connector_nodes],pc_by_individual_wcd[non_connector_nodes])
for i,p,n in zip(result[0],result[1],x_names):
	if p < 0.01:
		print i,p,n
result = scipy.stats.ttest_ind(pc_by_individual_bcd[connector_nodes],pc_by_individual_bcd[non_connector_nodes])
for i,p,n in zip(result[0],result[1],x_names):
	if p < 0.01:
		print i,p,n
	

mod_change_corr = np.zeros(subject_pcs.shape[1])
for i in range(subject_pcs.shape[1]):
	mod_change_corr[i] = pearsonr(subject_mods,subject_changes[:,i])[0]
#which nodes pc correlate with modularity?
mod_pc_corr = np.zeros(subject_pcs.shape[1])
for i in range(subject_pcs.shape[1]):
	mod_pc_corr[i] = pearsonr(subject_mods,subject_pcs[:,i])[0]
# which nodes number of changes correlate with modularity
mod_num_change_corr = np.zeros(subject_pcs.shape[1])
for i in range(subject_pcs.shape[1]):
	mod_num_change_corr[i] = pearsonr(subject_mods,subject_num_changes[:,i])[0]

print 'Modularity X Mean WMS: ' + str(pearsonr(subject_mods,subject_wms))
print 'Modularity X Mean BMS: ' + str(pearsonr(subject_mods,subject_bms))
print 'Modularity X WMS/BMS: ' + str(pearsonr(subject_mods,np.array(subject_wms)/np.array(subject_bms)))
print 'Modularity X Mean PC: ' + str(pearsonr(subject_mods,np.nanmean(subject_pcs,axis=1)))

"""
check motion
"""
# for subject in subjects:
#     e = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_12_False_engagement.npy' %(subject))
#     m = np.loadtxt('/home/despoB/mb3152/data/nki_data/preprocessed/pipeline_comp_cor_and_standard/%s_session_1/frame_wise_displacement/_scan_RfMRI_mx_645_rest/FD.1D'%(subject))
#     print pearsonr(m,np.std(e.reshape(900,12),axis=1)

