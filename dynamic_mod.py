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
from scipy.stats import ttest_ind
import glob
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

data_dir = '/home/despoB/mb3152/data/nki_data/preprocessed/pipeline_comp_cor_and_standard'
subject_dir = '%s/SUBJECT_session_1/functional_mni/_scan_RfMRI_mx_645_rest/_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96/_compcor_ncomponents_5_selector_pc10.linear1.wm1.global1.motion1.quadratic1.gm0.compcor0.csf1/_bandpass_freqs_0.009.0.08/**' %(data_dir)
hcp_subject_dir = '/home/despoB/connectome-data/SUBJECT/TASK/*reg*'
hcp_resting_dir = '/home/despoB/connectome-data/SUBJECT/*rfMRI*/*reg*'

subjects = ['0194023', '0185428', '0123657', '0141795', '0163508', '0123971', '0158411', '0185781', '0103714',
 '0174363', '0188854', '0136303', '0144667', '0139480', '0163228', '0154423', '0187635', '0179005', '0154555',
 '0150404', '0168357', '0192197', '0137496', '0159429', '0142673', '0180093', '0141860', '0116065', '0134795',
 '0150525', '0196198', '0162704', '0193358', '0105290', '0137073', '0112249', '0117168', '0125747', '0119947',
 '0138333', '0114688', '0167693', '0127665', '0117747', '0113436', '0127484', '0172267', '0152992', '0197584',
 '0176913', '0176479', '0188324', '0181179', '0159461', '0188757', '0102826', '0170400', '0112586', '0161348',
 '0114326', '0105409', '0105316', '0152366', '0157908']
hcp_subjects = os.listdir('/home/despoB/connectome-data/')

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

def run_component_estimation_hcp(subject,task,num_comps=12,ignore_flex=False):
	global subject_dir
	global hcp_subject_dir
	try:
		e = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_%s_%s_%s_engagement.npy'%(subject,num_comps,ignore_flex,task))
		e = e.reshape(-1,num_comps)
	except:
		component_file = get_2d_volume_data('/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Yeo_%sComp_PrActGivenComp_FSL_MNI152_2mm.nii.gz' %(num_comps))
		subject_path = hcp_subject_dir.replace('SUBJECT',str(subject))
		subject_path = subject_path.replace('TASK',task)
		epi_data = brain_graphs.load_subject_time_series(subject_path)
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
		np.save('/home/despoB/mb3152/dynamic_mod/component_activation/%s_%s_%s_%s_engagement.npy'%(subject,num_comps,ignore_flex,task), np.array(e))
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

def flex_activity(subject, num_comps = 12,ignore_flex=False,flex_thresh=3):
	flex = '/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Flexibility/YeoMD_%scomp_FSL_MNI152_thresh1e-5.nii' %(num_comps)
	flex = nib.load(flex).get_data().astype('float64')
	mask = nib.load('/usr/local/fsl-5.0.1/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz').get_data()
	components_engaged_var = []
	components_engaged_mean = []
	flex_activity = []
	non_flex_activity= []
	component_engagement = run_component_estimation(subject=subject,num_comps=num_comps,ignore_flex=ignore_flex)
	epi_data = brain_graphs.load_subject_time_series(subject_dir.replace('SUBJECT',str(subject)))
	epi_data[np.std(epi_data,axis=3)==0.0] = np.nan
	epi_data[mask<=0] = np.nan
	for i in range(epi_data.shape[-1]):
		brain_data = epi_data[:,:,:,i]
		non_flex_activity.append(np.nanmean(brain_data[flex<flex_thresh]))
		flex_activity.append(np.nanmean(brain_data[flex>=flex_thresh]))
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

def flex_activity_hcp(subject, task, num_comps = 12,ignore_flex=False,flex_thresh=3):
	flex = '/home/despoB/mb3152/modularity/YeoBrainmapMNI152/FSL/Flexibility/YeoMD_%scomp_FSL_MNI152_thresh1e-5.nii' %(num_comps)
	flex = nib.load(flex).get_data().astype('float64')
	mask = nib.load('/usr/local/fsl-5.0.1/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz').get_data()
	components_engaged_var = []
	components_engaged_mean = []
	flex_activity = []
	non_flex_activity= []
	component_engagement = run_component_estimation_hcp(subject=subject,num_comps=num_comps,ignore_flex=ignore_flex,task=task)
	subject_path = hcp_subject_dir.replace('SUBJECT',str(subject))
	subject_path = subject_path.replace('TASK',task)
	epi_data = brain_graphs.load_subject_time_series(subject_path)
	epi_data[np.std(epi_data,axis=3)==0.0] = np.nan
	epi_data[mask<=0] = np.nan
	for i in range(epi_data.shape[-1]):
		brain_data = epi_data[:,:,:,i]
		non_flex_activity.append(np.nanmean(brain_data[flex<flex_thresh]))
		flex_activity.append(np.nanmean(brain_data[flex>=flex_thresh]))
		engagement = np.array(component_engagement[i])
		components_engaged_var.append(1-np.std(engagement))
		components_engaged_mean.append(len(engagement[engagement>(1./float(num_comps))]))
	print pearsonr(components_engaged_var,flex_activity)
	print pearsonr(components_engaged_var,non_flex_activity)
	x = components_engaged_var
	y = components_engaged_mean
	z = flex_activity
	w = non_flex_activity
	np.save('/home/despoB/mb3152/dynamic_mod/component_activation/%s_flex_data_%s_%s_%s.npy'%(subject,num_comps,ignore_flex,task), np.array([x,y,z,w]))

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

def read_results_hcp(task,subjects=None,atlas='Shen',a_type='flex',corr_type='var',num_comps=12,ignore_flex=False,pc_thresh=8000,plot=False):
	if subjects == None:
		subjects = []
		if a_type == 'flex':
			subject_files = glob.glob('/home/despoB/mb3152/dynamic_mod/component_activation/**_%s_data_%s_%s_%s.npy' %(a_type,num_comps,ignore_flex,task))
			for subject_file in subject_files:
				subjects.append(subject_file.split('_')[2].split('/')[1])  		
		else:
			subject_files = glob.glob('/home/despoB/mb3152/dynamic_mod/component_activation/0**_%s_data_%s_%s_%s.npy' %(a_type,num_comps,ignore_flex,pc_thresh))
			for subject_file in subject_files:
				subjects.append(subject_file.split('_')[2].split('/')[1])  	   
	d = []
	for i in subject_files:
		d.append(list(np.load(i)))
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
		g = sns.regplot('Components Engaged','High Flexibility', df,color='Red',scatter=True,scatter_kws={'alpha':.1})
		# plt.xlim(min(df['Components Engaged']),max(df['Components Engaged']))
		plt.yticks(size=14)
		plt.xticks([])
		# plt.ylim(min(df['High Flexibility']),max(df['High Flexibility']))
		# plt.ylim(-100,100)
		plt.ylabel('Mean Activity at Connector Areas',size=24)
		plt.xlabel("Cognitive Components Engaged",size=24)
		sns.despine()
		plt.show()
		g = sns.regplot('Components Engaged','Low Flexibility', df,color='Blue',scatter_kws={'alpha':.1})
		# plt.xlim(min(df['Components Engaged']),max(df['Components Engaged']))
		# print min(df['Components Engaged'])
		# print max(df['Components Engaged'])
		# plt.ylim(min(df['Low Flexibility']),max(df['Low Flexibility']))
		plt.ylim(-50,50)
		plt.yticks(size=14)
		# plt.yticks([])
		plt.xticks([])
		plt.ylabel('Mean Activity at Local Areas',size=24)
		plt.xlabel("Cognitive Components Engaged",size=24)
		sns.despine()
		plt.show()

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

def task_performance(subjects,task):
	all_performance = []
	for subject in subjects:
		try:
			df = pd.read_csv('/home/despoB/mb3152/scanner_performance_data/%s_tfMRI_%s_Stats.csv' %(subject,task))
		except:
			all_performance.append(np.nan)
			continue
		if task[:-3] == 'WM':
			performance = np.mean(df['Value'][[24,27,30,33]])
		if task[:-3] == 'RELATIONAL':
			performance = df['Value'][1]
		if task[:-3] == 'LANGUAGE':
			performance = df['Value'][1]
		if task[:-3] == 'SOCIAL':
			performance = np.mean([df['Value'][0],df['Value'][5]])
		all_performance.append(performance)
	return np.array(all_performance)

def edges_task_performance(subjects,task,atlas):
	matrix = []
	analyze_subjects = []
	for subject in subjects:
		try:
			matrix.append(np.load('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_tfMRI_%s_matrix.npy'%(subject,atlas,task)))
			analyze_subjects.append(subject)
		except:
			continue
	matrix = np.array(matrix)
	matrix[np.isnan(matrix)] = 0.0
	num_nodes = matrix.shape[-1]
	result = np.zeros((num_nodes,num_nodes))
	tp = task_performance(analyze_subjects,task)
	matrix = np.delete(matrix,np.argwhere(np.isnan(tp)).reshape(-1),axis=0)
	tp = tp[np.isnan(tp)==False]
	for i1,i2, in combinations(range(num_nodes),2):
		val = pearsonr(matrix[:,i1,i2],tp)[0]
		result[i1,i2] = val
		result[i2,i1] = val
	return result

def edges_all_performance(subjects,atlas,tasks=['WM_LR','RELATIONAL_LR','LANGUAGE_LR','SOCIAL_LR']):
	mean = []
	for task in tasks:
		mean.append(edges_task_performance(subjects,task,atlas))
	return np.array(mean)

def plot_corr_matrix(matrix,membership):
	swap_dict = {}
	index = 0
	corr_mat = np.zeros((matrix.shape))
	names = []
	x_ticks = []
	y_ticks = []
	for i in np.unique(membership):
		for node in np.where(membership==i)[0]:
			swap_dict[node] = index
			index = index + 1
			names.append(membership[node])
	y_names = []
	x_names = []
	old_name = 0
	for i,name in enumerate(names):
		if name == old_name:
			continue
		old_name = name
		y_ticks.append(i)
		x_ticks.append(len(names)-i)
		y_names.append(name)
		x_names.append(name)
	for i in range(len(swap_dict)):
		for j in range(len(swap_dict)):
			corr_mat[swap_dict[i],swap_dict[j]] = matrix[i,j]
			corr_mat[swap_dict[j],swap_dict[i]] = matrix[j,i]
	sns.set(context="paper", font="monospace")
	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(12, 9))
	# Draw the heatmap using seaborn
	y_names.reverse()
	sns.heatmap(corr_mat,square=True,yticklabels=y_names,xticklabels=x_names,linewidths=0.0,)
	ax.set_yticks(x_ticks)
	ax.set_xticks(y_ticks)
	membership.sort()
	# Use matplotlib directly to emphasize known networks
	for i, network in enumerate(membership):
		if network != membership[i - 1]:
			ax.axhline(len(membership) - i, c='black',linewidth=2)
			ax.axvline(i, c='black',linewidth=2)
	f.tight_layout()
	plt.show()

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

def individual_graph_analyes(variables):
	subject = variables[0]
	print subject
	atlas = variables[1]
	task = variables[2]
	files = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_*%s*_matrix.npy'%(subject,atlas,task))
	s_matrix = []
	for f in files:
		print 'loaded' + f
		s_matrix.append(np.load(f))
	s_matrix = np.nanmean(s_matrix,axis=0)
	s_matrix[np.isnan(s_matrix)] = 0.0
	np.fill_diagonal(s_matrix,0.0)
	pc = []
	mod = []
	wmd = []
	for cost in np.array(range(5,16))*0.01:
		temp_matrix = s_matrix.copy()
		graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,check_tri=True,interpolation='midpoint',normalize=True)
		del temp_matrix
		graph = graph.community_infomap(edge_weights='weight')
		graph = brain_graphs.brain_graph(graph)
		pc.append(np.array(graph.pc))
		wmd.append(np.array(graph.wmd))
		mod.append(graph.community.modularity)
		del graph
	return (mod,np.nanmean(pc,axis=0),np.nanmean(wmd,axis=0))

def run_price(project,subject,task,ignore_flex=3):
	if project == 'hcp':
		run_component_estimation_hcp(subject=subject,task=task,ignore_flex=ignore_flex)
		flex_activity_hcp(subject=subject,task=task,ignore_flex=ignore_flex)
	else:
		run_component_estimation(subject,ignore_flex=ignore_flex)
		flex_activity(subject,ignore_flex=ignore_flex)

def make_static_matrix(subject,task,project,atlas):
	parcel_path = '/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas)
	if project == 'nki':
		subject_path = subject_dir.replace('SUBJECT',subject)
	if project == 'hcp':
		subject_path = hcp_resting_dir.replace('SUBJECT',subject)
		subject_path = subject_path.replace('*rfMRI*',task)
	if project == 'hcp_task':
		subject_path = hcp_subject_dir.replace('SUBJECT',subject)
		subject_path = subject_path.replace('TASK',task)
	subject_time_series = brain_graphs.load_subject_time_series(subject_path)
	matrix = brain_graphs.time_series_to_matrix(subject_time_series,parcel_path,voxel=False,fisher=False,out_file=None)
	np.save('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_matrix.npy' %(subject,atlas,task),matrix)

def multi_slice(subject,task,project,atlas='power',gamma=1.0,omega=.1,cost=0.1,window_size=100):
	parcel_path = '/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas)
	if project == 'nki':
		subject_path = subject_dir.replace('SUBJECT',subject)
	if project == 'hcp':
		subject_path = hcp_resting_dir.replace('SUBJECT',subject)
		subject_path = subject_path.replace('*rfMRI*',task)
	if project == 'hcp_task':
		subject_path = hcp_subject_dir.replace('SUBJECT',subject)
		subject_path = subject_path.replace('TASK',task)
	subject_time_series = brain_graphs.load_subject_time_series(subject_path)
	matrix = brain_graphs.time_series_to_matrix(subject_time_series,parcel_path,voxel=False,fisher=False,out_file=None)
	np.save('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_matrix.npy'%(subject,atlas,task),matrix)
	matrix = brain_graphs.time_series_to_ewmf_matrix(subject_time_series=subject_time_series,parcel_path=parcel_path,window_size=window_size,out_file=None)
	del subject_time_series
	out_file = '/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_%s_msc_%s_%s_%s.npy' %(subject,atlas,window_size,cost,gamma,omega,task)
	brain_graphs.multi_slice_community(matrix=matrix,cost=cost,out_file=out_file,omega=omega,gamma=gamma)

def get_project(subjects):
	if subjects[0][0] == '0':
		project = 'nki'
	else:
		project = 'hcp'
	return project

def graph_metrics(subjects,task,atlas):
	"""
	run graph metrics or load them
	"""
	variables = []
	project = get_project(subjects)
	for subject in subjects:
		files = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_*%s*_matrix.npy'%(subject,atlas,task))
		if len(files) == 0:
			subjects.remove(subject)
			continue
		variables.append([subject,atlas,task])
	try:
		subject_pcs = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_pcs.npy' %(project,task))
		subject_wmds = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_wmds.npy' %(project,task))
		subject_mods = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_mods.npy'%(project,task))
	except:
		subject_mods = [] #individual subject modularity values
		subject_pcs = [] #subjects PCs
		print 'Running Graph Theory Analyses'
		from multiprocessing import Pool
		pool = Pool(20)
		results = pool.map(individual_graph_analyes,variables)
		subject_pcs = []
		subject_mods = []
		subject_wmds = []
		for r in results:
			subject_mods.append(np.nanmean(r[0]))
			subject_pcs.append(r[1])
			subject_wmds.append(r[2])
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_pcs.npy' %(project,task),np.array(subject_pcs))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_wmds.npy' %(project,task),np.array(subject_wmds))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_mods.npy' %(project,task),np.array(subject_mods))
	matrices = []
	thresh_matrices = []
	for subject in subjects:
		s_matrix = []
		files = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_*%s*_matrix.npy'%(subject,atlas,task))
		for f in files:
			s_matrix.append(np.load(f))
		s_matrix = np.nanmean(s_matrix,axis=0)
		s_matrix[np.isnan(s_matrix)] = 0.0
		np.fill_diagonal(s_matrix,0.0)
		num_nodes = s_matrix.shape[0]
		thresh_matrices.append(scipy.stats.zscore(s_matrix.reshape(-1)).reshape((num_nodes,num_nodes)))
		matrices.append(s_matrix)
		variables.append([subject,atlas,task])	
	subject_mods = np.array(subject_mods)
	subject_pcs = np.array(subject_pcs)
	subject_wmds = np.array(subject_wmds)
	matrices = np.array(matrices)
	thresh_matrices = np.array(thresh_matrices)
	results = {}
	results['subject_pcs'] = subject_pcs
	results['subject_mods'] = subject_mods
	results['subject_wmds'] = subject_wmds
	results['matrices'] = matrices
	del matrices
	results['thresh_matrices'] = thresh_matrices
	del thresh_matrices
	return results

def dynamic_graph_metrics(subjects,task,atlas='power',window_size=100,msc_cost=0.1,gamma=1.0,omega=0.1):
	"""
	multi slice stuff
	"""
	project = get_project(subjects)
	try:
		1/0
		r = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_%s_%s_%s_%s_msc.npy'%(project,task,atlas,window_size,msc_cost,gamma,omega))
		results = {}
		results['subject_changes'] = r[0]
		results['subject_num_changes'] = r[1]
	except:
		subject_changes = [] #communities at each node
		subject_num_changes = [] #number of changes at a node
		for subject in subjects:
			files = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_%s_msc_%s_%s_*%s*.npy' %(subject,atlas,window_size,msc_cost,gamma,omega,task))
			msc = []
			for f in files:
				print f
				msc.append(np.load(f))
			msc = np.array(msc)
			msc = msc.reshape(-1,msc.shape[-1])
			num_communities = []
			for i in range(msc.shape[1]):
				num_communities.append(len(np.unique(msc[:,i])))
			subject_changes.append(num_communities)
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
			subject_num_changes.append(num_changes)
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_%s_%s_%s_%s_msc.npy'%(project,task,atlas,window_size,msc_cost,gamma,omega),np.array([subject_changes,subject_num_changes]))
		results = {}
		shape = (len(subject_changes),len(subject_changes[0]))
		print shape
		subject_changes_array = np.zeros(shape)
		subject_num_changes_array = np.zeros(shape)
		for i in range(shape[0]):
			subject_changes_array[i] = subject_changes[i]
			subject_num_changes_array[i] = subject_num_changes[i]
		results['subject_changes'] = subject_changes_array
		results['subject_num_changes'] = subject_num_changes_array
	return results

def edge_weight_and_performance(task,subjects=hcp_subjects,atlas='Power',mean=False):
	if mean == True:
		mean = edges_all_performance(subjects,atlas)
	else:
		mean = edges_task_performance(subjects,atlas,task)
	project = get_project(subjects)
	graph_metrics = graph_metrics(subjects,task)
	pc = np.nanmean(graph_metrics['subject_pcs'],axis=0)
	connectors = np.argwhere(pc>=thresh).reshape(-1)
	non_connectors = np.argwhere(pc<thresh).reshape(-1)
	matrix = np.nansum(np.absolute(mean),axis=0)
	print thresh, scipy.stats.ttest_ind(matrix[np.ix_(connectors,non_connectors)].reshape(-1),matrix[np.ix_(non_connectors,non_connectors)].reshape(-1))
	print thresh, scipy.stats.ttest_ind(matrix[np.ix_(connectors,connectors)].reshape(-1),matrix[np.ix_(non_connectors,non_connectors)].reshape(-1))
	sns.violinplot([matrix[np.ix_(connectors,connectors)].reshape(-1),matrix[np.ix_(connectors,non_connectors)].reshape(-1),matrix[np.ix_(non_connectors,non_connectors)].reshape(-1)])
	plt.show()

def check_motion(subjects):
	for subject in subjects:
	    e = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_12_False_engagement.npy' %(subject))
	    m = np.loadtxt('/home/despoB/mb3152/data/nki_data/preprocessed/pipeline_comp_cor_and_standard/%s_session_1/frame_wise_displacement/_scan_RfMRI_mx_645_rest/FD.1D'%(subject))
	    print pearsonr(m,np.std(e.reshape(900,12),axis=1))

def pc_edge_correlation(subject_pcs,matrices,path):
	try:
		pc_edge_corr = np.load(path)
	except:
		pc_edge_corr = np.zeros((subject_pcs.shape[1],subject_pcs.shape[1],subject_pcs.shape[1]))
		subject_pcs[np.isnan(subject_pcs)] = 0.0
		for i in range(subject_pcs.shape[1]):
			for n1,n2 in combinations(range(subject_pcs.shape[1]),2):
				val = pearsonr(subject_pcs[:,i],matrices[:,n1,n2])[0]
				pc_edge_corr[i,n1,n2] = val
				pc_edge_corr[i,n2,n1] = val
		np.save(path,pc_edge_corr)
	return pc_edge_corr

def remove_missing_subjects(subjects,task,atlas='power',gamma=1.0,omega=0.1,msc_cost = 0.1,window_size=100):
	for subject in subjects:
		files = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_%s_msc_%s_%s_*%s*.npy' %(subject,atlas,window_size,msc_cost,gamma,omega,task))
		if len(files) == 0.0:
			subjects.remove(subject)
			continue
		files = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_*%s*_matrix.npy'%(subject,atlas,task))
		if len(files) == 0.0:
			subjects.remove(subject)
	return subjects

def main_analyes(task,subjects=hcp_subjects,project='hcp',atlas='power',gamma=1.0,omega=0.1,msc_cost = 0.1,window_size=100,pc_thresh=75):
	task = 'REST'
	subjects=hcp_subjects
	project='hcp'
	atlas='power'
	gamma=1.0
	omega=0.1
	msc_cost = 0.1
	window_size=100
	pc_thresh=75

	subjects = remove_missing_subjects(subjects,task,atlas,gamma,omega,msc_cost,window_size)
	known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
	known_membership[known_membership==-1] = 0
	colors = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[34].values)
	network_names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)
	num_nodes = len(known_membership)
	name_int_dict = {}
	for name,int_value in zip(network_names,known_membership):
		name_int_dict[int_value] = name
	"""
	Make a matrix of each nodes PC correlation to all edges in the graph.
	"""
	static_results = graph_metrics(subjects,task,atlas)
	dynamic_results = dynamic_graph_metrics(subjects,task)
	subject_pcs = static_results['subject_pcs']
	thresh_matrices = static_results['thresh_matrices']
	subject_mods = static_results['subject_mods']
	subject_changes = dynamic_results['subject_changes']
	subject_num_changes = dynamic_results['subject_num_changes']
	pc_edge_corr = pc_edge_correlation(subject_pcs,thresh_matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr.npy' %(project,task,atlas))
	pc_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),pc_thresh)
	connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)>=pc_thresh)[0]
	non_connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)<pc_thresh)[0]
	low_pc_edge_matrix = np.nanmean(pc_edge_corr[non_connector_nodes],axis=0)
	high_pc_edge_matrix = np.nanmean(pc_edge_corr[connector_nodes],axis=0)
	matrix = np.nansum([scipy.tril(low_pc_edge_matrix),scipy.triu(high_pc_edge_matrix)],axis=0)
	plot_corr_matrix(matrix,network_names.copy())
	#community averaged weight changes. This is not just an average of above, but looking at average connectivity between networks and PC
	# bcd_pc_network_matrix_mod = np.zeros((num_nodes,len(np.unique(known_membership)),len(np.unique(known_membership))))
	# for n in range(num_nodes):
	# 	for c1,c2, in combinations(range(len(np.unique(known_membership))),2):
	# 		community_nodes = np.where(known_membership==c1)[0]
	# 		non_community_nodes = np.where(known_membership==c2)[0]
	# 		bcds = []
	# 		for s in range(thresh_matrices.shape[0]):
	# 			bcds.append(np.nanmean(thresh_matrices[s][np.ix_(community_nodes,non_community_nodes)]))
	# 		bcd = pearsonr(subject_pcs[:,n],bcds)[0]
	# 		bcd_pc_network_matrix_mod[n,c1,c2] = bcd
	# 		bcd_pc_network_matrix_mod[n,c2,c1] = bcd
	# low_pc_network_matrix = np.nanmean(bcd_pc_network_matrix_mod[non_connector_nodes],axis=0)
	# high_pc_network_matrix = np.nanmean(bcd_pc_network_matrix_mod[connector_nodes],axis=0)
	# sns.heatmap(np.nansum([scipy.tril(low_pc_network_matrix),scipy.triu(high_pc_network_matrix)],axis=0),square=True,yticklabels=x_names,xticklabels=x_names,linewidths=0.0,)
	# plt.tight_layout()
	# plt.yticks(size=16)
	# plt.xticks(size=16)
	# plt.show()
	"""
	analyze that matrix (modulation analyses)
	"""
	#Within and between network edge PC modulation weights in matrix for each network, for each node.
	community_mod_high_wcd = np.zeros((len(connector_nodes),len(np.unique(known_membership))))
	community_mod_low_wcd = np.zeros((len(non_connector_nodes),len(np.unique(known_membership))))
	community_mod_high_bcd = np.zeros((len(connector_nodes),len(np.unique(known_membership))))
	community_mod_low_bcd = np.zeros((len(non_connector_nodes),len(np.unique(known_membership))))
	community_mod_bcd = np.zeros((num_nodes,len(np.unique(known_membership))))
	community_mod_wcd = np.zeros((num_nodes,len(np.unique(known_membership))))
	community_mod_high_ratio = np.zeros((len(connector_nodes),len(np.unique(known_membership))))
	community_mod_low_ratio = np.zeros((len(non_connector_nodes),len(np.unique(known_membership))))
	for i,n in enumerate(connector_nodes):
		for community in np.unique(known_membership):
			community_nodes = np.where(known_membership==community)[0]
			non_community_nodes = np.where(known_membership!=community)[0]
			wcd = float(np.nanmean(pc_edge_corr[n][np.ix_(community_nodes,community_nodes)]))
			bcd = float(np.nanmean(pc_edge_corr[n][np.ix_(non_community_nodes,community_nodes)]))
			community_mod_high_wcd[i][community] = wcd
			community_mod_high_bcd[i][community] = bcd
			community_mod_high_ratio[i][community] = wcd/bcd
	for i,n in enumerate(non_connector_nodes):
		for community in np.unique(known_membership):
			community_nodes = np.where(known_membership==community)[0]
			non_community_nodes = np.where(known_membership!=community)[0]
			wcd = float(np.nanmean(pc_edge_corr[n][np.ix_(community_nodes,community_nodes)]))
			bcd = float(np.nanmean(pc_edge_corr[n][np.ix_(non_community_nodes,community_nodes)]))
			community_mod_low_wcd[i][community] = wcd
			community_mod_low_bcd[i][community] = bcd
			community_mod_low_ratio[i][community] = wcd/bcd
	for i,n in enumerate(range(num_nodes)):
		for community in np.unique(known_membership):
			community_nodes = np.where(known_membership==community)[0]
			non_community_nodes = np.where(known_membership!=community)[0]
			wcd = float(np.nanmean(pc_edge_corr[n][np.ix_(community_nodes,community_nodes)]))
			bcd = float(np.nanmean(pc_edge_corr[n][np.ix_(non_community_nodes,community_nodes)]))
			community_mod_wcd[i][community] = wcd
			community_mod_bcd[i][community] = bcd

	#PC is correlated positively with wcd and negatively with bcd
	print 'ttest_ind: within community degree modulation, between community degree modulation'
	print scipy.stats.ttest_ind(community_mod_wcd.reshape(-1),community_mod_bcd.reshape(-1))
	#we can look at this relationship for connector nodes or non-connector nodes. Stronger at connector nodes for wcd, stronger at non for bcd
	#between community degree modulation is a decrease for connector nodes, increase for non_connector nodes.
	print 'ttest_ind: connectors within community degree modulation, non_connectors within community degree modulation'
	print scipy.stats.ttest_ind(community_mod_high_wcd.reshape(-1),community_mod_low_wcd.reshape(-1))
	#within community degree modulation is an increase for connector nodes, decrease for non_connector nodes.
	print 'ttest_ind: connectors between community degree modulation, non_connectors between community degree modulation'
	print scipy.stats.ttest_ind(community_mod_high_bcd.reshape(-1),community_mod_low_bcd.reshape(-1))

	#test if some networks are more driven by connector nodes than non-connector nodes.
	print 'within community degree is increased more than between community degree'
	for community in np.unique(known_membership):
		print name_int_dict[community], scipy.stats.ttest_ind(community_mod_wcd[:,community].reshape(-1),community_mod_bcd[:,community].reshape(-1))
	print 'within community degree is increased more by connector nodes than non_connector nodes'
	for community in np.unique(known_membership):
		print name_int_dict[community], scipy.stats.ttest_ind(community_mod_high_wcd[:,community].reshape(-1),community_mod_low_wcd[:,community].reshape(-1))
	print 'between community degree is decreased more by connector nodes than non_connector nodes'
	for community in np.unique(known_membership):
		print name_int_dict[community], scipy.stats.ttest_ind(community_mod_high_bcd[:,community].reshape(-1),community_mod_low_bcd[:,community].reshape(-1))
	print 'ratio of community degree to between community degree is increased more by connector nodes than non_connector nodes'
	for community in np.unique(known_membership):
		print name_int_dict[community], scipy.stats.ttest_ind(community_mod_high_ratio[:,community].reshape(-1),community_mod_low_ratio[:,community].reshape(-1))

	#make brain plot of networks that have a wcd that is more strongly shaped by 
	network_change_corr = np.zeros(subject_pcs.shape[1])
	for community in np.unique(known_membership):
		community_nodes = np.where(known_membership==community)[0]
		wcd_sum = []
		for n in connector_nodes:
			wcd_sum.append(np.mean(pc_edge_corr[n][np.ix_(community_nodes,community_nodes)].reshape(-1)))
		network_change_corr[community==known_membership] = scipy.stats.ttest_ind(community_high_wcd[:,community].reshape(-1),community_low_wcd[:,community].reshape(-1))[0]

	df = pd.read_csv('/home/despoB/mb3152/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
	df[3] = network_change_corr
	df.to_csv('power_mod_change_connector_wcd.node',sep='\t',index=False,names=False,header=False)

	#plot box plots of the values each communities modulation within and between.
	violin_pc_edge_corr_bcd = []
	violin_pc_edge_corr_wcd = []
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
			violin_pc_edge_corr_wcd[community].extend(wcd[wcd!=0])
			violin_pc_edge_corr_bcd[community].extend(bcd[bcd!=0])
	#plot the within community degree by community.
	df = pd.DataFrame()
	for i,name in enumerate(x_names):
		if i == 0:
			df = pd.DataFrame(violin_pc_edge_corr_wcd[i],columns=[x_names[i]])
			continue
		temp_df = pd.DataFrame(violin_pc_edge_corr_wcd[i],columns=[x_names[i]])
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
	#plot the between community degree by community.
	df = pd.DataFrame()
	for i,name in enumerate(x_names):
		if i == 0:
			df = pd.DataFrame(violin_pc_edge_corr_bcd[i],columns=[x_names[i]])
			continue
		temp_df = pd.DataFrame(violin_pc_edge_corr_bcd[i],columns=[x_names[i]])
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
	"""
	specificity of modulation
	"""
	#sum of weight changes for each node, by each node.
	weight_change_matrix = np.zeros((num_nodes,num_nodes))
	weight_change_matrix_pos = np.zeros((num_nodes,num_nodes))
	weight_change_matrix_neg = np.zeros((num_nodes,num_nodes))
	for n1 in range(num_nodes):
		for n2 in range(num_nodes):
			array = pc_edge_corr[n1][n2]
			weight_change_matrix[n1,n2] = np.sum(np.absolute(array))
			weight_change_matrix_pos[n1,n2] = np.sum(array[array>0])
			weight_change_matrix_neg[n1,n2] = np.sum(array[array<0])

	#correlate sum of negative weights by pc edge weight.
	sns.regplot(weight_change_matrix_neg.reshape(-1),np.nanmean(thresh_matrices,axis=0).reshape(-1),color='Blue',scatter=True,scatter_kws={'alpha':.15})
	plt.xlabel('Sum of negative pc modulation changes',size=24)
	plt.ylabel('Edge weight between nodes',size=24)
	plt.yticks(size=16)
	plt.xticks(size=16)
	plt.show()
	print pearsonr(weight_change_matrix_neg.reshape(-1),np.nanmean(thresh_matrices,axis=0).reshape(-1))

	sns.regplot(weight_change_matrix_pos.reshape(-1),np.nanmean(thresh_matrices,axis=0).reshape(-1),color='Red',scatter=True,scatter_kws={'alpha':.15})
	plt.xlabel('Sum of postive pc modulation changes',size=24)
	plt.ylabel('Edge weight between nodes',size=24)
	plt.yticks(size=16)
	plt.xticks(size=16)
	plt.show()
	print pearsonr(weight_change_matrix_pos.reshape(-1),np.nanmean(thresh_matrices,axis=0).reshape(-1))

	sns.regplot(weight_change_matrix.reshape(-1),np.nanmean(thresh_matrices,axis=0).reshape(-1),color='Red',scatter=True,scatter_kws={'alpha':.15})
	plt.xlabel('Sum of all modulation changes',size=24)
	plt.ylabel('Edge weight between nodes',size=24)
	plt.yticks(size=16)
	plt.xticks(size=16)
	plt.show()
	print pearsonr(weight_change_matrix.reshape(-1),np.nanmean(thresh_matrices,axis=0).reshape(-1))

	#Correlation of each node's PC with bcd and wcd of each network, across subjects.
	pc_by_individual_wcd = np.zeros((num_nodes,len(np.unique(known_membership))))
	pc_by_individual_bcd = np.zeros((num_nodes,len(np.unique(known_membership))))
	pc_by_individual_mod = np.zeros((num_nodes,len(np.unique(known_membership))))
	for ix,i in enumerate(range(num_nodes)):
		for c,name in zip(range(len(np.unique(known_membership))),names):
			pc_by_individual_wcd[ix,c] = pearsonr(subject_pcs[:,i],community_wcd[:,c])[0]
			pc_by_individual_bcd[ix,c] = pearsonr(subject_pcs[:,i],community_bcd[:,c])[0]
			pc_by_individual_mod[ix,c] = pearsonr(subject_pcs[:,i],community_wcd[:,c]/community_bcd[:,c])[0]

	#last test of specifity, have not run yet! !! ! 
	#Mean connectivity of each node to each network,relative to overall strength
	#modulation of a network, realtive to overall modulation values.
	community_stregth = np.zeros((len(subjects),num_nodes,len(np.unique(known_membership))))
	community_mod_stregth = np.zeros((len(subjects),num_nodes,len(np.unique(known_membership))))
	for i,subject in enumerate(subjects):
		for n in range(num_nodes):
			for community in np.unique(known_membership):
				community_nodes = np.where(known_membership==community)[0]
				community_stregth[i,n,community] = (np.nanmean(thresh_matrices[i,n,community_nodes])/np.nanmean(thresh_matrices[i,n,:]))**2
				community_mod_stregth[i,n,community] = (np.nanmean(pc_edge_corr[i,n,community_nodes])/np.nanmean(pc_edge_corr[i,n,:]))**2

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
	
	#need matrix of node to community strength across subjects for this.
	"""
	which nodes pc correlate with modularity?
	"""
	mod_change_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_change_corr[i] = pearsonr(subject_mods,subject_changes[:,i])[0]

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

	sns.regplot(subject_mods,np.nanmean(subject_pcs,axis=1))
	plt.yticks(size=16)
	plt.xlabel('Modularity',size=16)
	plt.xticks(size=16)
	plt.ylabel('Mean Participation Coefficient',size=16)
	plt.show()

	subject_num_changes = np.array(subject_num_changes)
	network_change_corr = np.zeros(subject_pcs.shape[1])
	for community in np.unique(known_membership):
		network_change_corr[community==known_membership] = scipy.stats.ttest_ind(community_mod_high_ratio[:,community].reshape(-1),community_mod_low_ratio[:,community].reshape(-1))[0]

	brain_graphs.make_image('/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas),'/home/despoB/mb3152/dynamic_mod/brain_figures/mod_change_corr_hcp_%s' %(atlas),mod_change_corr*100)

"""
SGE Inputs
"""

if len(sys.argv) > 1:
	if sys.argv[1] == 'multislice':
		subject = sys.argv[2]
		task = sys.argv[3]
		if 'nki' in task:
			project = 'nki'
			task = 'mb_rest_645'
		if 'rfMRI' in task:
			project = 'hcp'
		if 'tfMRI' in task:
			project = 'hcp_task'
		print subject, task , project
		multi_slice(subject,task,project)
	if sys.argv[1] == 'flex_activity':
		subject = sys.argv[2]
		flex_activity(subject=subject,ignore_flex=False)
	if sys.argv[1] == 'flex_activity_hcp':
		subject = sys.argv[2]
		task = sys.argv[3]
		flex_activity_hcp(subject=subject,task=task,ignore_flex=False)
	if sys.argv[1] == 'forever':
		a = 0
		while True:
			a = a - 1
			a = a + 1



"""
Methods

Pre-processing
HCP, minimally pre-processed, + WM, CSF, Global
NKI, standard with GSR to match HCP

1.	Resting-State Matrix
	a.	HCP, do all, average
	b.	NKI, MB .645
2.	EWMC, MSM
	a.	HCP task LR and RL
	b.	HCP rest
	c.	NKI, MB .645
3.	Component Estimates
	a.	HCP Rest, LR1, LR2
	b.	HCP Task, LR
	c.	NKI, MB .645

Results
1.	Correlations between components engaged (1-variance).
	a.	NKI Rest Data--flex_activity()
	b.	HCP Rest Data--flex_activity_hcp()
	c.	HCP Task Data--flex_activity_hcp(task)
		i.	Task frames only?
2.	Nodes that change the most, have high PC, predict higher modularity.
	a.	NKI Rest Data
	b.	HCP Rest Data
	c.	HCP Task Data
		i.	Brain Map for each task.
		ii.	Do nodes that increase PC in a single task also increase community changes?
3. Correlation matrix figure connector nodes PC positively correlates with between module decreases.
	a.	NKI Rest Data
	b.	HCP Rest Data
		i.	Correlation between all PC and modularity is weak.
4.	Correlation between PC values/nodal changes and performance?
	a.	HCP Task data
	b.	Do correlation between rest changes and behavioral measures, same nodes?
5.	Correlation between edge weights and task performance, are these connector node edges?
	a.	HCP Task Data
	b.	Do correlation between rest edge weights and behavioral measures? Same edges?
6.	Rich Connector Club.
	a.	Looks like using the PC cutoff gets the same nodes?
7.	In multi-slice modularity, is there an increase in BOLD magnitude in the connector regions (perhaps defined statically) during shifts to the right (more PC)? 
	Does each regions PC during multislice correlate with it's activity? Maybe only connector nodes?
"""
# if atlas =='gordon':
# 	df = pd.read_excel('/home/despoB/mb3152/dynamic_mod/Parcels.xlsx')
# 	names = df.Community
# 	known_membership = np.zeros(len(names))
# 	for i,c in enumerate(np.unique(names)):
# 		known_membership[df.ParcelID[df.Community==c]-1]=i
# 	names[names=='None'] = 'Uncertain'