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

subjects = ['0194023',
 '0185428',
 '0123657',
 '0141795',
 '0163508',
 '0123971',
 '0158411',
 '0185781',
 '0103714',
 '0174363',
 '0188854',
 '0136303',
 '0144667',
 '0139480',
 '0163228',
 '0154423',
 '0187635',
 '0179005',
 '0154555',
 '0150404',
 '0168357',
 '0192197',
 '0137496',
 '0159429',
 '0142673',
 '0180093',
 '0141860',
 '0116065',
 '0134795',
 '0150525',
 '0196198',
 '0162704',
 '0193358',
 '0105290',
 '0137073',
 '0112249',
 '0117168',
 '0125747',
 '0119947',
 '0138333',
 '0114688',
 '0167693',
 '0127665',
 '0117747',
 '0113436',
 '0127484',
 '0172267',
 '0152992',
 '0197584',
 '0176913',
 '0176479',
 '0188324',
 '0181179',
 '0159461',
 '0188757',
 '0102826',
 '0170400',
 '0112586',
 '0161348',
 '0114326',
 '0105409',
 '0105316',
 '0152366',
 '0157908']

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

def dynamic_graph_analyses(subjects=None,atlas='shen',msc_cost=0.1,hub_cost = .1,num_comps=12,ignore_flex=4,window_size=100,c_method='resursive'):
	atlas = 'shen'
	gamma = 1.0
	omega = .1
	msc_cost = 0.1
	num_comps = 12
	window_size = 100
	hub_cost = .1
	c_method = 'avg_0.01_0.1'
	if subjects == None:
		subjects = []
		subject_paths = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/*_%s_%s_%s_msc_%s_%s.npy' %(atlas,window_size,msc_cost,gamma,omega))
		for s in subject_paths:
			subjects.append(s.split('/')[-1].split('_')[0])
	subject_mods = [] #individual subject modularity values
	subject_changes = [] # communities at each node
	subject_num_changes = [] # number of changes at a node
	subject_pcs = [] # subjects PCs
	subject_bms = [] #between module strength
	subject_wms = [] #within module strength
	group_pc = [] #group PC
	group_graph = brain_graphs.load_graph('/home/despoB/mb3152/dynamic_mod/graphs/%s_%s'%(atlas,c_method))
	# cost = .25
	matrix = []
	for subject in subjects:
		matrix.append(np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_matrix.npy'%(subject,atlas)))
	# matrix = np.nanmean(matrix,axis=0)
	# while True:
	# 	graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost)
	# 	group_pc.append(brain_graphs.brain_graph(VertexClustering(graph,partition_graph.community.membership)).pc)
	# 	cost = cost - 0.001
	# 	if cost < .05:
	# 		break
	# group_pc = np.nanmean(group_pc,axis=0)

	for subject in subjects:
		print subject
		msc = np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_%s_msc_%s_%s.npy' %(subject,atlas,window_size,msc_cost,gamma,omega))
		# matrix = np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_ewmf.npy' %(subject,atlas,window_size))
		# try:
		# 	mods = np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_msc_mods_%s.npy'%(subject,atlas,window_size,msc_cost))
		# 	subject_mods.append(mods)
		# except:
		# 	mods = []
		# 	for i in range(msc.shape[0]):
		# 		graph = brain_graphs.matrix_to_igraph(matrix[i,:,:],hub_cost)
		# 		mods.append(VertexClustering(graph, membership=msc[i].astype(int)).modularity)
		# 	np.save('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_msc_mods_%s'%(subject,atlas,window_size,hub_cost,msc_cost),np.array(mods))
		s_matrix = np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_matrix.npy'%(subject,atlas))
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
			cost = cost - 0.005
			if cost < .05:
				break
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
	#which nodes changes correlate with modularity?
	mod_change_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_change_corr[i] = pearsonr(subject_mods,subject_changes[:,i])[0]
	#which nodes pc correlate with modularity?
	mod_pc_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_pc_corr[i] = pearsonr(subject_mods,subject_pcs[:,i])[0]

	mod_num_change_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_num_change_corr[i] = pearsonr(subject_mods,subject_num_changes[:,i])[0]

	print 'Modularity X Mean WMS: ' + str(pearsonr(subject_mods,subject_wms))
	print 'Modularity X Mean BMS: ' + str(pearsonr(subject_mods,subject_bms))
	print 'Modularity X WMS/BMS: ' + str(pearsonr(subject_mods,np.array(subject_wms)/np.array(subject_bms)))
	print 'Modularity X Mean PC: ' + str(pearsonr(subject_mods,np.nanmean(subject_pcs,axis=1)))

	df = pd.DataFrame(data=np.array([subject_mods,np.nanmean(subject_pcs,axis=1)]).transpose(),columns=['Modularity','Mean Participation Coefficient'])
	sns.regplot('Modularity','Mean Participation Coefficient',df,scatter=True)
	plt.show()
	brain_graphs.make_image('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas),'/home/despoB/mb3152/dynamic_mod/brain_figures/mod_num_change_corr_%s' %(atlas),mod_num_change_corr*10000)
	brain_graphs.make_image('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas),'/home/despoB/mb3152/dynamic_mod/brain_figures/mod_change_corr_%s' %(atlas),mod_change_corr*10000)
	brain_graphs.make_image('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas),'/home/despoB/mb3152/dynamic_mod/brain_figures/mod_pc_corr_%s' %(atlas),mod_pc_corr*10000)
	brain_graphs.make_image('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas),'/home/despoB/mb3152/dynamic_mod/brain_figures/pc_%s' %(atlas),np.nanmean(subject_pcs,axis=0)*10000)

	subject_num_changes = np.array(subject_num_changes)
	mod_num_change_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_num_change_corr[i] = pearsonr(subject_mods,subject_num_changes[:,i])[0]
	brain_graphs.make_image('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas),'/home/despoB/mb3152/dynamic_mod/brain_figures/mod_num_change_corr_%s' %(atlas),mod_num_change_corr*10000)

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


# looks as connector nodes increase degree and provicinal hubs decrease
# does pc * components engaged work for connector HUBS?

#both dc and fc from specific timepoints show no modularity differences. but huge one from total partition to high components partition
#but when compared to the whole time series it is. but, is this just from there being less data? randomly select 225 data points

#when two components are engaged, is there stronger connectivity between them?

#when a component is engaged does it have higher connectivity to the rest of the brain or itself?

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
# atlas = 'craddock_280'
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




"""
check motion
"""
# for subject in subjects:
#     e = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_12_False_engagement.npy' %(subject))
#     m = np.loadtxt('/home/despoB/mb3152/data/nki_data/preprocessed/pipeline_comp_cor_and_standard/%s_session_1/frame_wise_displacement/_scan_RfMRI_mx_645_rest/FD.1D'%(subject))
#     print pearsonr(m,np.std(e.reshape(900,12),axis=1)


# atlas = 'gordon'
# subject = str(sys.argv[1])
# num_comps = 12
# ignore_flex = 4
# window_size = 100
# cost = 0.05
# engagement = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_%s_%s_engagement.npy'%(subject,num_comps,ignore_flex))
# engagement = 1-np.array(np.std(engagement.reshape(engagement.shape[0],num_comps),axis=1))
# division = np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_%s_msc.npy' %(subject,atlas,window_size,cost))
# matrix = np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_ewmf.npy' %(subject,atlas,window_size))
# mods = []
# pcs = []
# cost = 0.1
# for tp in range(900-window_size*2):
# 	graph = brain_graphs.matrix_to_igraph(matrix[window_size + tp],cost)
# 	membership = division[tp]
# 	p = brain_graphs.brain_graph(VertexClustering(graph, membership=membership.astype('int8')))
# 	mods.append(graph.modularity(membership=abs(membership.astype(int))))
# 	pcs.append(np.nanmean(p.pc))
# print pearsonr(mods,engagement[window_size:-window_size])
# print pearsonr(pcs,engagement[window_size:-window_size])



# num_comps = 12
# ignore_flex = 3.5
# length = 50
# subject = subjects[0]


# # # subject_path = subject_dir.replace('SUBJECT',subject)
# # # subject_time_series = brain_graphs.load_subject_time_series(subject_path)
# # # rand_subject_time_series = subject_time_series.copy()
# # # orig_shape = subject_time_series.shape
# # # rand_subject_time_series = rand_subject_time_series.reshape(-1)
# # # np.random.shuffle(rand_subject_time_series)
# # # rand_subject_time_series = rand_subject_time_series.reshape(orig_shape)

# rand_subject_time_series = np.random.rand(91,109,91,length)
# mask = nib.load('/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Yeo_%sComp_PrActGivenComp_FSL_MNI152_2mm.nii.gz' %(num_comps))
# mask = mask.get_data()
# rand_subject_time_series[np.max(mask,axis=3)==0.0] = 0.0
# flex = '/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Flexibility/YeoMD_%scomp_FSL_MNI152_thresh1e-5.nii' %(num_comps)
# flex = nib.load(flex).get_data().astype('float64')

# # mask = nib.load('/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Yeo_%sComp_PrActGivenComp_FSL_MNI152_2mm.nii.gz' %(num_comps))
# # mask_data = mask.get_data()
# # mask_data[:,:,:,:] = rand_subject_time_series[:,:,:,10:22]
# # nib.save(mask,'test.nii')


# epi_data = rand_subject_time_series.copy()
# component_file = get_2d_volume_data('/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Yeo_%sComp_PrActGivenComp_FSL_MNI152_2mm.nii.gz' %(num_comps))

# if ignore_flex != False:
# 	rand_subject_time_series[flex>=ignore_flex] = 0.0
# components_engaged_var = []
# flex_activity = []
# non_flex_activity= []
# epi_data[np.max(epi_data,axis=3)==0.] = np.nan
# c_values = []
# for i in range(length):
# 	brain_data = rand_subject_time_series[:,:,:,i]
# 	brain_data = brain_data.reshape(902629,1)
# 	e = estimate(brain_data,component_file)
# 	c_values.append(e[0])
# 	print np.mean(c_values,axis=0)
# 	brain_data = epi_data[:,:,:,i]
# 	non_flex_activity.append(np.nanmean(brain_data[flex<ignore_flex]))
# 	flex_activity.append(np.nanmean(brain_data[flex>=ignore_flex]))
# 	components_engaged_var.append(1-np.std(e))
# 	print pearsonr(components_engaged_var,flex_activity)
# 	print pearsonr(components_engaged_var,non_flex_activity)

# engagement = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_%s_%s_engagement.npy'%(subjects[50],num_comps,ignore_flex))
# real_engagement = 1-np.array(np.std(engagement.reshape(engagement.shape[0],num_comps),axis=1))[:900]

# print pearsonr(real_engagement[:length],components_engaged_var[:900])
"""
find timepoints 
"""


# pc_activity(subject= str(sys.argv[1]),num_comps = 12,ignore_flex=False)
# from scipy.stats import pearsonr
# import numpy as np
# num_comps = 12
# window = 100

# scores = []
# for subject in subjects:
# 	c = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_12_False_engagement.npy'%(subject))
# 	scores.append(np.sum(np.max(c,axis=0)))
# subject = subjects[np.argmax(scores)]
# # component_image = nib.load('/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Yeo_12Comp_PrActGivenComp_FSL_MNI152_2mm.nii.gz')
# # component_data = component_image.get_data()
# c = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_12_False_engagement.npy'%(subject))
# global c

# from multiprocessing import Pool
# from moviepy.editor import *

# def write_image(i):
# 	global c
# 	print i
# 	plt.figure(figsize=(11.75,7.5))
# 	plt.plot(c[i:i+50,0,:],lw=3)
# 	plt.yticks([0.,.2,.4,.6])
# 	plt.xticks([])
# 	plt.savefig('/tmp/%07d_image.png' %(i),dpi=150,orientation='landscape' )
# 	plt.close()

# pool = Pool(processes=12)         
# pool.map(write_image, range(300))
# files = glob.glob('/tmp/*image.png*')
# files.sort()
# # os.system("ffmpeg -f image2 -i '/tmp/%07d_image.png' estimate_video.mpg")
# clip = ImageSequenceClip(files,fps=10)
# clip.write_videofile(filename='estimates_300.mp4',codec='mpeg4',bitrate='max',audio=False,threads=12)

# scores = []
# window = 100
# for i in range(900):
# 	if i > 849:
# 		continue
# 	scores.append(np.std(np.max(c[i:i+window,0,:],axis=0)))
# for component_num in range(12):
# 	i = np.argmin(scores)
# 	print component_num
# 	component = component_data[:,:,:,component_num].copy()
# 	component[component<1] = 0.0
# 	shape = component_data.shape
# 	component_final = np.zeros((shape[0],shape[1],shape[2],window))
# 	for ti,i in zip(range(window),range(i,i+window)):
# 	    component_final[:,:,:,ti] = component * c[i][0][component_num] * 100
# 	new_img = nib.Nifti1Image(component_final, component_image.affine, component_image.header)
# 	nib.save(new_img, "/home/despoB/mb3152/dynamic_mod/brain_figures/component_%s_%s.nii.gz" %(component_num,subject))
# subject_path = subject_dir.replace('SUBJECT',subject)
# epi = brain_graphs.load_subject_time_series(subject_path)
# i = np.argmin(scores)
# new_img = nib.Nifti1Image(epi[:,:,:,i:i+window].astype('float16'), component_image.affine, component_image.header)
# nib.save(new_img, "/home/despoB/mb3152/dynamic_mod/brain_figures/epi_%s.nii.gz" %(subject))

# for subject in subjects:
# 	if subject == '0158411':
# 		continue
# 	if subject != '0141795':
# 		continue
# 	c = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_12_4_engagement.npy'%(subject))
# 	window = 15
# 	time = np.linspace(0,window,window)
# 	for i1 in range(c.shape[0]):
# 		if i1 < window:
# 			continue
# 		if i1 > c.shape[0] - window - 1:
# 			continue
# 		i2 = i1 + window
# 		comps = c[i1:i2]
# 		x = np.zeros(num_comps)
# 		for i in range(num_comps):
# 			x[i] = pearsonr(comps[:,0,i],time)[0]
# 		if len(x[x>.95])==2:
# 			if len(x[x<.45])==10:
# 				if len(x[x>-.45])==10:
# 					component_nums = np.where([x>.9])[1]
# 					component_nums = np.insert(component_nums,2,9)
# 					for component_num in component_nums:
# 						data =c[i1:i2,:,component_num]
# 						component_image = nib.load('/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Yeo_12Comp_PrActGivenComp_FSL_MNI152_2mm.nii.gz')
# 						component_data = component_image.get_data()
# 						component = component_data[:,:,:,component_num].copy()
# 						component[component<1] = 0.0
# 						shape = component_data.shape
# 						component_data = np.zeros((shape[0],shape[1],shape[2],window))
# 						for i in range(window):
# 						    component_data[:,:,:,i] = component[:,:,:] * data[i] * 100
# 						new_img = nib.Nifti1Image(component_data, component_image.affine, component_image.header)
# 						nib.save(new_img, "/home/despoB/mb3152/dynamic_mod/brain_figures/component_%s_%s.nii.gz" %(component_num,subject))
# 					subject_path = subject_dir.replace('SUBJECT',subject)
# 					epi = brain_graphs.load_subject_time_series(subject_path)
# 					new_img = nib.Nifti1Image(epi[:,:,:,i1:i2], component_image.affine, component_image.header)
# 					nib.save(new_img, "/home/despoB/mb3152/dynamic_mod/brain_figures/epi_%s.nii.gz" %(subject))
# 					1/0
