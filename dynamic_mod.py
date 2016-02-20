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
from itertools import combinations, permutations
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
from richclub import preserve_strength, RC
from sklearn import linear_model, cross_validation, svm
import time
from multiprocessing import Pool
#build graphs for timepoints when component is engaged.
#build graphs for when no variance versus high variance, look at modularity and PC and WMD. Perhaps calculate 
#modularity without PC nodes / See if most of the between module connections come from PC nodes.

hcp_subject_dir = '/home/despoB/connectome-data/SUBJECT/*TASK*/*reg*'
hcp_resting_dir = '/home/despoB/connectome-data/SUBJECT/*TASK*/*reg*'
hcp_subjects = os.listdir('/home/despoB/connectome-data/')
hcp_subjects.sort()

def nan_pearsonr(x,y):
	x = np.array(x)
	y = np.array(y)
	isnan = np.sum([x,y],axis=0)
	isnan = np.isnan(isnan) == False
	return pearsonr(x[isnan],y[isnan])

def entropy(s):
	p, lns = Counter(s), float(len(s))
	return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def task_performance(subjects,task):
	all_performance = []
	bdf = pd.read_csv('/home/despoB/mb3152/dynamic_mod/os_behavior_data.csv')	
	for subject in subjects:
		try:
			files = glob.glob('/home/despoB/mb3152/scanner_performance_data/%s_tfMRI_*%s*_Stats.csv' %(subject,task))
			performance = []
			for f in files:
				df = pd.read_csv(f)
				if task == 'WM':
					t_performance = np.mean(df['Value'][[24,27,30,33]])
					# s1 = bdf['ListSort_AgeAdj'][bdf.Subject == int(subject)]
					# t_performance = np.nanmean([t_performance,s1])
				if task == 'RELATIONAL':
					t_performance = np.mean([df['Value'][0],df['Value'][1]])
					# s1 = bdf['PMAT24_A_CR'][bdf.Subject == int(subject)]
					# t_performance = np.nanmean([t_performance,s1])
				if task == 'LANGUAGE':
					t_performance = np.mean([df['Value'][2],df['Value'][5]])
					s1 = bdf['ReadEng_AgeAdj'][bdf.Subject == int(subject)] 
					s2 = bdf['PicVocab_AgeAdj'][bdf.Subject == int(subject)]
					t_performance = np.nanmean([t_performance,s1,s2])
				if task == 'SOCIAL':
					t_performance = np.mean([df['Value'][0],df['Value'][5]])
					# s1 = bdf['ER40_CR'][bdf.Subject == int(subject)]
					# t_performance = np.nanmean([t_performance,s1]) 
				performance.append(t_performance)
			all_performance.append(np.mean(performance))
		except:
			all_performance.append(np.nan)
	return np.array(all_performance)

def behavioral_performance(subjects,tasks):
	"""
	which edges correlate with performance?
	--Possible Tasks--
	PMAT24_A_CR: Penn Matrix Test: Median Reaction Time for Correct Responses. More Better.
	PicSeq_AgeAdj: Episodic Memory (Picture Sequence Memory). More Better.
	CardSort_AgeAdj: Executive Function/Cognitive Flexibility (Dimensional Change Card Sort). More Better.
	Flanker_AgeAdj: Executive Function/Inhibition (Flanker Task). More Better.
	ReadEng_AgeAdj: Language/Reading Decoding (Oral Reading Recognition). More Better.
	PicVocab_AgeAdj: Language/Vocabulary Comprehension (Picture Vocabulary). More Better.
	ProcSpeed_AgeAdj: Processing Speed (Pattern Completion Processing Speed). More Better.
	DDisc_AUC_200, DDisc_AUC_40K: Delay Discounting: Area Under the Curve for Discounting of XXX. Less Better.
	VSPLOT_CRTE: Variable Short Penn Line Orientation: Median Reaction Time Divided by Expected Number of Clicks for Correct. Less Better.
	SCPT_SPEC: Short Penn Continuous Performance Test: Specificity = SCPT_TN/(SCPT_TN + SCPT_FP). More Better.
	SCPT_SEN: Short Penn Continuous Performance Test: Sensitivity = SCPT_TP/(SCPT_TP + SCPT_FN). More Better.
	IWRD_TOT: Penn Word Memory Test:  Total Number of Correct Responses (IWRD_TOT). More Better.
	ListSort_AgeAdj: Working Memory (List Sorting). More Better.
	"""
	bdf = pd.read_csv('/home/despoB/mb3152/dynamic_mod/os_behavior_data.csv')
	bdf['VSPLOT_CRTE'] = bdf['VSPLOT_CRTE'] * -1
	bdf['DDisc_AUC_200'] = bdf['DDisc_AUC_200'] * -1
	bdf['DDisc_AUC_40K'] = bdf['DDisc_AUC_40K'] * -1
	results = np.zeros((len(subjects),len(tasks)))
	for id_s,subject in enumerate(subjects):
		for id_t, task in enumerate(tasks):
			r=bdf[task][bdf.Subject == int(subject)]
			if len(r) > 0:
				results[id_s,id_t] = bdf[task][bdf.Subject == int(subject)]
			else:
				results[id_s,id_t] = np.nan
	return results

def edges_task_performance(subjects,task,atlas):
	matrix = []
	for subject in subjects:
		subject_matrix = []
		for f in glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_tfMRI_*%s*_matrix.npy'%(subject,atlas,task)):
			f = np.load(f)
			f[np.isnan(f)] = 0.0
			np.fill_diagonal(f,0.0)
			num_nodes = f.shape[-1]
			f = scipy.stats.zscore(f.reshape(-1)).reshape((num_nodes,num_nodes))
			subject_matrix.append(f)		
		matrix.append(np.nanmean(subject_matrix,axis=0))
	matrix = np.array(matrix)
	num_nodes = matrix.shape[-1]
	result = np.zeros((num_nodes,num_nodes))
	tp = task_performance(subjects,task)
	matrix = np.delete(matrix,np.argwhere(np.isnan(tp)).reshape(-1),axis=0)
	tp = tp[np.isnan(tp)==False]
	for i1,i2, in combinations(range(num_nodes),2):
		val = pearsonr(matrix[:,i1,i2],tp)[0]
		result[i1,i2] = val
		result[i2,i1] = val
	return result

def edges_all_performance(subjects,atlas,tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL']):
	mean = []
	for task in tasks:
		mean.append(edges_task_performance(subjects,task,atlas))
	return np.array(mean)

def plot_corr_matrix(matrix,membership,out_file=None,block_lower=False,return_array=True,plot_corr=True):
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
	if block_lower:
		corr_mat = scipy.triu(corr_mat)
	sns.heatmap(corr_mat,square=True,yticklabels=y_names,xticklabels=x_names,linewidths=0.0,cmap="coolwarm")
	ax.set_yticks(x_ticks)
	ax.set_xticks(y_ticks)
	membership.sort()
	# Use matplotlib directly to emphasize known networks
	for i, network in enumerate(membership):
		if network != membership[i - 1]:
			ax.axhline(len(membership) - i, c='black',linewidth=2)
			ax.axvline(i, c='black',linewidth=2)
	f.tight_layout()
	if out_file != None:
		plt.savefig(out_file,dpi=1200)
		plt.close()
	if plot_corr == True:
		plt.show()
	if return_array == True:
		plt.close()
		return corr_mat

def get_power_pc(hub='pc'):
    data = pd.read_csv('/home/despoB/mb3152/modularity/mmc3.csv')
    data['new'] = np.zeros(len(data))
    if hub == 'pc':
        for x2,y2,z2,pc in zip(data.X2,data.Y2,data.Z2,data.PC):
            data.new[data[data.X1==x2][data.Z1==z2][data.Y1==y2].index[0]] = pc
    else:
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
	print files
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
	return (mod,np.nanmean(pc,axis=0),np.nanmean(wmd,axis=0),subject)

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

def run_multi_slice(subject,task,project,atlas='power',gamma=1.0,omega=.1,cost=0.1,window_size=100):
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

def dynamic_graph_metrics(subjects,task,atlas='power',project='hcp',window_size=100,msc_cost=0.1,gamma=1.0,omega=0.1):
	"""
	multi slice stuff
	"""
	if atlas == 'power':
		num_nodes = 264
	try:
		r = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_%s_%s_%s_%s_msc.npy'%(project,task,atlas,window_size,msc_cost,gamma,omega))
		results = {}
		results['subject_changes'] = r
	except:
		subject_changes = np.zeros((len(subjects),num_nodes)) #communities at each node
		for s_ix,subject in enumerate(subjects):
			files = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_%s_msc_%s_%s_*%s*.npy' %(subject,atlas,window_size,msc_cost,gamma,omega,task))
			for f in files:
				print f
				msc = np.load(f)
				for i in range(msc.shape[1]):
					changes = len(np.unique(msc[:,i]))
					subject_changes[s_ix][i] = subject_changes[s_ix][i] + changes
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_%s_%s_%s_%s_msc.npy'%(project,task,atlas,window_size,msc_cost,gamma,omega),subject_changes)
		results = {}
		results['subject_changes'] = subject_changes
	return results

def individual_graph_analyes(variables):
	subject = variables[0]
	print subject
	atlas = variables[1]
	task = variables[2]
	s_matrix = variables[3]
	pc = []
	mod = []
	wmd = []
	for cost in np.array(range(50,250))*0.001:
	# for cost in np.array(range(5,16))*0.01:
		temp_matrix = s_matrix.copy()
		graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True)
		del temp_matrix
		graph = graph.community_infomap(edge_weights='weight')
		graph = brain_graphs.brain_graph(graph)
		pc.append(np.array(graph.pc))
		wmd.append(np.array(graph.wmd))
		mod.append(graph.community.modularity)
		del graph
	return (mod,np.nanmean(pc,axis=0),np.nanmean(wmd,axis=0),subject)

def graph_metrics(subjects,task,atlas,project = 'hcp'):
	"""
	run graph metrics or load them
	"""
	try:
		done_subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %(project,task,atlas)) 
		assert (done_subjects == subjects).all() #make sure you are getting subjects / subjects order you wanted and ran last time.
		subject_pcs = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pcs_fz.npy' %(project,task,atlas))
		subject_wmds = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_wmds_fz.npy' %(project,task,atlas))
		subject_mods = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_mods_fz.npy'%(project,task,atlas))
		matrices = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_matrices_fz.npy'%(project,task,atlas))
		thresh_matrices = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_z_matrices_fz.npy'%(project,task,atlas))
	except:
		variables = []
		matrices = []
		thresh_matrices = []
		for subject in subjects:
			s_matrix = []
			files = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_*%s*_matrix.npy'%(subject,atlas,task))
			for f in files:
				f = np.load(f)
				np.fill_diagonal(f,0.0)
				f[np.isnan(f)] = 0.0
				f = np.arctanh(f)
				s_matrix.append(f.copy())
			s_matrix = np.nanmean(s_matrix,axis=0)
			variables.append([subject,atlas,task,s_matrix.copy()])
			num_nodes = s_matrix.shape[0]
			thresh_matrix = s_matrix.copy()
			thresh_matrix = scipy.stats.zscore(thresh_matrix.reshape(-1)).reshape((num_nodes,num_nodes))
			thresh_matrices.append(thresh_matrix.copy())
			matrices.append(s_matrix.copy())
		subject_mods = [] #individual subject modularity values
		subject_pcs = [] #subjects PCs
		subject_wmds = []
		print 'Running Graph Theory Analyses'
		from multiprocessing import Pool
		pool = Pool(5)
		results = pool.map(individual_graph_analyes,variables)		
		for r,s in zip(results,subjects):
			subject_mods.append(np.nanmean(r[0]))
			subject_pcs.append(r[1])
			subject_wmds.append(r[2])
			assert r[3] == s #make sure it returned the order of subjects/results correctly
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pcs_fz_0.05_0.25.npy' %(project,task,atlas),np.array(subject_pcs))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_wmds_fz_0.05_0.25.npy' %(project,task,atlas),np.array(subject_wmds))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_mods_fz_0.05_0.25.npy' %(project,task,atlas),np.array(subject_mods))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz_0.05_0.25.npy' %(project,task,atlas),np.array(subjects))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_matrices_fz_0.05_0.25.npy'%(project,task,atlas),np.array(matrices))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_z_matrices_fz_0.05_0.25.npy'%(project,task,atlas),np.array(thresh_matrices))
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
	results['z_scored_matrices'] = thresh_matrices
	results['subjects'] = subjects
	del thresh_matrices
	return results

def edge_weight_and_performance(tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],subjects=hcp_subjects,project='hcp',atlas='power',mean=False,thresh=75):
	mean = []
	abs_mean = []
	for task in tasks:
		subjects = np.array(hcp_subjects).copy()
		subjects = remove_missing_subjects(subjects,task,atlas)
		perf_matrix = edges_task_performance(subjects,task,atlas)
		static_metrics = graph_metrics(subjects,task,atlas)
		pc = np.nanmean(static_metrics['subject_pcs'],axis=0)
		abs_matrix = np.absolute(perf_matrix.copy())
		matrix = perf_matrix.copy()
		matrix[np.isnan(matrix)] = 0.0
		abs_matrix[np.isnan(abs_matrix)] = 0.0
		pc_thresh = np.percentile(pc,75)
		local_thresh = np.percentile(pc,25)
		connectors = np.where(pc>=pc_thresh)[0]
		non_connectors = np.where(pc<local_thresh)[0]
		mean.append(matrix)
		abs_mean.append(abs_matrix)
		print task
		print 'weighted'
		print thresh, scipy.stats.ttest_ind(matrix[np.ix_(connectors,non_connectors)].reshape(-1),matrix[np.ix_(non_connectors,non_connectors)].reshape(-1))
		print thresh, scipy.stats.ttest_ind(matrix[np.ix_(connectors,connectors)].reshape(-1),matrix[np.ix_(non_connectors,non_connectors)].reshape(-1))
		print 'absolute'
		print thresh, scipy.stats.ttest_ind(abs_matrix[np.ix_(connectors,non_connectors)].reshape(-1),abs_matrix[np.ix_(non_connectors,non_connectors)].reshape(-1))
		print thresh, scipy.stats.ttest_ind(abs_matrix[np.ix_(connectors,connectors)].reshape(-1),abs_matrix[np.ix_(non_connectors,non_connectors)].reshape(-1))
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

def remove_missing_subjects(subjects,task,atlas='power'):
	"""
	remove missing subjects, original array is being edited
	"""
	subjects = list(subjects)
	for subject in subjects:
		# files = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_%s_%s_msc_%s_%s_*%s*.npy' %(subject,atlas,window_size,msc_cost,gamma,omega,task))
		# if len(files) == 0.0:
		# 	subjects.remove(subject)
		# 	continue
		# if len(np.load(files[0]).shape) < 2:
		# 	subjects.remove(subject)
		# 	continue
		files = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_*%s*_matrix.npy'%(subject,atlas,task))
		if len(files) < 2:
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
	subjects = remove_missing_subjects(subjects,task,atlas,gamma,omega,msc_cost,window_size)
	known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
	known_membership[known_membership==-1] = 0
	colors = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[34].values)
	network_names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)
	num_nodes = len(known_membership)
	name_int_dict = {}
	color_int_dict = {}
	for name,color,int_value in zip(network_names,colors,known_membership):
		name_int_dict[int_value] = name
		color_int_dict[int_value] = color
	"""
	run graph theory analyses
	"""
	static_results = graph_metrics(subjects,task,atlas)
	dynamic_results = dynamic_graph_metrics(subjects,task)
	subject_pcs = static_results['subject_pcs']
	thresh_matrices = static_results['thresh_matrices']
	subject_mods = static_results['subject_mods']
	subject_changes = dynamic_results['subject_changes']

	"""
	which nodes pc and module changes correlate with modularity?
	"""
	subject_pcs[np.isnan(subject_pcs)] = 0.0
	mod_change_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_change_corr[i] = pearsonr(subject_mods,subject_changes[:,i])[0]
	mod_pc_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_pc_corr[i] = pearsonr(subject_mods,subject_pcs[:,i])[0]
	print 'PC x (PC and modularity):' + str(pearsonr(mod_change_corr,np.nanmean(subject_pcs,axis=0)))
	print 'PC x (Changes and modularity): ' + str(pearsonr(mod_pc_corr,np.nanmean(subject_pcs,axis=0)))
	
	# df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
	# df[4] = 5.5
	# if task == 'REST':
	# 	df[3] = np.nanmean(subject_pcs,axis=0)
	# 	df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_static_pc.node',sep='\t',index=False,names=False,header=False)
	# maxv = np.mean(mod_pc_corr) + (np.std(mod_pc_corr)*2)
	# minv = np.mean(mod_pc_corr) - (np.std(mod_pc_corr)*2)	
	# df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
	# df[3] = mod_pc_corr
	# df[3][mod_pc_corr > maxv] = maxv
	# df[3][mod_pc_corr < minv] = minv
	# maxv = np.mean(mod_change_corr) + (np.std(mod_change_corr)*2)
	# minv = np.mean(mod_change_corr) - (np.std(mod_change_corr)*2)
	# df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_mod_%s.node'%(task),sep='\t',index=False,names=False,header=False)
	# df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
	# df[3] = mod_change_corr
	# df[3][mod_change_corr > maxv] = maxv
	# df[3][mod_change_corr < minv] = minv
	# df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_ms_mod_%s.node'%(task),sep='\t',index=False,names=False,header=False)
	"""
	Make a matrix of each node's PC correlation to all edges in the graph.
	"""
	pc_thresh = 75
	pc_edge_corr = pc_edge_correlation(subject_pcs,thresh_matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr.npy' %(project,task,atlas))
	pc_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),pc_thresh)
	connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)>=pc_thresh)[0]
	non_connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)<pc_thresh)[0]
	low_pc_edge_matrix = np.nanmean(pc_edge_corr[non_connector_nodes],axis=0)
	high_pc_edge_matrix = np.nanmean(pc_edge_corr[connector_nodes],axis=0)
	matrix = np.nansum([scipy.tril(low_pc_edge_matrix),scipy.triu(high_pc_edge_matrix)],axis=0)
	# plot_corr_matrix(matrix,network_names.copy())

	"""
	average pc_edge_corr by network
	"""
	network_pc_edge_corr = np.zeros((14,14))
	for n1,n2 in combinations(range(14),2):
		n1_nodes = np.where(known_membership==n1)[0].reshape(-1)
		n2_nodes = np.where(known_membership==n2)[0].reshape(-1)
		network_pc_edge_corr[n1,n2] = np.nanmean(high_pc_edge_matrix[np.ix_(n1_nodes,n2_nodes)])
		network_pc_edge_corr[n2,n1] = np.nanmean(low_pc_edge_matrix[np.ix_(n1_nodes,n2_nodes)])
	sns.heatmap(network_pc_edge_corr,square=True,yticklabels=name_int_dict.values(),xticklabels=name_int_dict.values())
	np.save(network_pc_edge_corr)


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

	#do connector nodes modify more edges than non-connector nodes?
	print scipy.stats.ttest_ind(np.array([community_mod_high_wcd.reshape(-1),community_mod_high_bcd.reshape(-1)]).reshape(-1),np.array([community_mod_low_wcd.reshape(-1),community_mod_low_bcd.reshape(-1)]).reshape(-1))

	print 'ttest_ind: connectors within community degree modulation, connectors between community degree modulation'
	print scipy.stats.ttest_ind(community_mod_high_wcd.reshape(-1),community_mod_high_bcd.reshape(-1))
	# print 'ttest_ind: non-connectors within community degree modulation, non-connectors between community degree modulation'
	# print scipy.stats.ttest_ind(community_mod_low_wcd.reshape(-1),community_mod_low_bcd.reshape(-1))
	# #PC is correlated positively with wcd and negatively with bcd
	# print 'ttest_ind: within community degree modulation, between community degree modulation'
	# print scipy.stats.ttest_ind(community_mod_wcd.reshape(-1),community_mod_bcd.reshape(-1))
	#we can look at this relationship for connector nodes or non-connector nodes. Stronger at connector nodes for wcd, stronger at non for bcd
	#between community degree modulation is a decrease for connector nodes, increase for non_connector nodes.
	print 'ttest_ind: connectors within community degree modulation, non_connectors within community degree modulation'
	print scipy.stats.ttest_ind(community_mod_high_wcd.reshape(-1),community_mod_low_wcd.reshape(-1))
	#within community degree modulation is an increase for connector nodes, decrease for non_connector nodes.
	print 'ttest_ind: connectors between community degree modulation, non_connectors between community degree modulation'
	print scipy.stats.ttest_ind(community_mod_high_bcd.reshape(-1),community_mod_low_bcd.reshape(-1))

	#test if some networks are more driven by connector nodes than non-connector nodes.
	# print 'within community degree is increased more than between community degree'
	# for community in np.unique(known_membership):
	# 	print name_int_dict[community], scipy.stats.ttest_ind(community_mod_wcd[:,community].reshape(-1),community_mod_bcd[:,community].reshape(-1))
	print 'within community degree is increased more than between community degree by connector nodes'
	for community in np.unique(known_membership):
		print name_int_dict[community], scipy.stats.ttest_ind(community_mod_high_wcd[:,community].reshape(-1),community_mod_high_bcd[:,community].reshape(-1))
	print 'within community degree is increased more by connector nodes than non_connector nodes'
	for community in np.unique(known_membership):
		print name_int_dict[community], scipy.stats.ttest_ind(community_mod_high_wcd[:,community].reshape(-1),community_mod_low_wcd[:,community].reshape(-1))
	# print 'between community degree is decreased more by connector nodes than non_connector nodes'
	# for community in np.unique(known_membership):
	# 	print name_int_dict[community], scipy.stats.ttest_ind(community_mod_high_bcd[:,community].reshape(-1),community_mod_low_bcd[:,community].reshape(-1))
	# print 'ratio of community degree to between community degree is increased more by connector nodes than non_connector nodes'
	# for community in np.unique(known_membership):
	# 	print name_int_dict[community], scipy.stats.ttest_ind(community_mod_high_ratio[:,community].reshape(-1),community_mod_low_ratio[:,community].reshape(-1))

	results = []
	temp_df = []
	for community in np.unique(known_membership):
		if scipy.stats.ttest_ind(community_mod_high_wcd[:,community].reshape(-1),community_mod_high_bcd[:,community].reshape(-1))[1] > 0.05:
			results.append('Null')
		if scipy.stats.ttest_ind(community_mod_high_wcd[:,community].reshape(-1),community_mod_high_bcd[:,community].reshape(-1))[0] > 0.0:
			results.append('Within Community Edges Increased')
		if scipy.stats.ttest_ind(community_mod_high_wcd[:,community].reshape(-1),community_mod_high_bcd[:,community].reshape(-1))[0] < 0.0:
			results.append('Between Community Edges Increased')
	for community,r in zip(np.unique(known_membership),results):
		for weight in community_mod_high_wcd[:,community].reshape(-1):
			temp_df.append([weight,'Within Community',r,name_int_dict[community]])
		for weight in community_mod_high_bcd[:,community].reshape(-1):
			temp_df.append([weight,'Between Communities',r,name_int_dict[community]])
	df = pd.DataFrame(temp_df,columns=['Connector Modulation','Edge Type','Result','Network'])
	sns.violinplot(x=df['Network'],y=df['Connector Modulation'], hue=df["Edge Type"],split=True,scale='width',inner="quartile",palette="Set3")
	plt.xticks(rotation=90)
	plt.yticks([-.2,-.1,0.,0.1,.2])
	plt.tight_layout()
	plt.show()
	1/0

	#plot box plots of the values each communities modulation within and between.
	violin_pc_edge_corr_bcd = []
	violin_pc_edge_corr_wcd = []
	for i in range(len(np.unique(known_membership))):
		violin_pc_edge_corr_bcd.append([])
		violin_pc_edge_corr_wcd.append([])
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
	for i in np.unique(known_membership.astype(int)):
		if i == 0:
			df = pd.DataFrame(violin_pc_edge_corr_wcd[i],columns=[name_int_dict[i]])
			continue
		temp_df = pd.DataFrame(violin_pc_edge_corr_wcd[i],columns=[name_int_dict[i]])
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
	for i in np.unique(known_membership.astype(int)):
		if i == 0:
			df = pd.DataFrame(violin_pc_edge_corr_bcd[i],columns=[name_int_dict[i]])
			continue
		temp_df = pd.DataFrame(violin_pc_edge_corr_bcd[i],columns=[name_int_dict[i]])
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
	specificity of modulation by nodes' pc?
	check to see if correlation is only there for connector nodes
	also, do absolute valies of the correaltion! 
	is it stronger during task than rest?
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
	matrices = static_results['matrices']
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
	print pearsonr(weight_change_matrix.reshape(-1),np.absolute(np.nanmean(thresh_matrices,axis=0).reshape(-1)))
	for i in range(14):
		print name_int_dict[i], pearsonr(community_mod_stregth[connector_nodes,i].reshape(-1),np.mean(community_stregth,axis=0)[connector_nodes,i].reshape(-1))
	for i in range(14):
		print name_int_dict[i], pearsonr(community_mod_stregth[non_connector_nodes,i].reshape(-1),np.mean(community_stregth,axis=0)[non_connector_nodes,i].reshape(-1))


	"""
	rich club stuff
	"""
	cost = 0.2
	graph = brain_graphs.matrix_to_igraph(np.nanmean(static_results['matrices'],axis=0),cost=cost)
	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	average_randomized_phis = np.mean([RC(preserve_strength(graph),scores=graph.strength(weights='weight')).phis() for i in range(500)])
	degree_normalized_phis = degree_emperical_phis/average_randomized_phis
	graph = brain_graphs.matrix_to_igraph(np.nanmean(static_results['matrices'],axis=0),cost=cost)
	pc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight')).pc
	pc[np.isnan(pc)] = 0.0
	pc_emperical_phis = RC(graph, scores=pc).phis()
	pc_average_randomized_phis = np.mean([RC(preserve_strength(graph),scores=pc).phis() for i in range(500)])
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	plt.plot(pc_normalized_phis,color='b',linestyle='-',label='PC')
	plt.plot(degree_normalized_phis,color='r',linestyle='-',label='Degree')
	# plt.plot(pc_emperical_phis,color='b')
	# plt.plot(degree_emperical_phis,color='r')
	plt.legend()
	plt.ylabel('Normalized Rich Club Coefficient')
	plt.xlabel('PC/Degree Rank')
	plt.show()
	plt.show()

def connectivity_across_tasks(subjects=hcp_subjects):
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
	project='hcp'
	atlas='power'
	known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
	known_membership[known_membership==-1] = 0
	colors = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[34].values)
	network_names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)
	num_nodes = len(known_membership)
	name_int_dict = {}
	color_int_dict = {}
	for name,color,int_value in zip(network_names,colors,known_membership):
		name_int_dict[int_value] = name
		color_int_dict[int_value] = color
	df_columns=['Participation Coefficient','Task','Diversity Facilitated Modularity Change']
	df = pd.DataFrame(columns = df_columns)
	for task in tasks:
		print task
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		subject_mods = static_results['subject_mods']
		task_perf = task_performance(subjects,task)
		assert subject_pcs.shape[0] == len(subjects)
		mean_pc = np.nanmean(subject_pcs,axis=0)
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
		print nan_pearsonr(mod_pc_corr,mean_pc)
		df_array = []
		for node in range(264):
			df_array.append([mean_pc[node],task,mod_pc_corr[node]])
		df = pd.concat([df,pd.DataFrame(df_array,columns=df_columns)],axis=0)
		predict_nodes = np.arange(len(mod_pc_corr))[np.argsort(mod_pc_corr)][-25:]
		local_predict_nodes = np.arange(len(mod_pc_corr))[np.argsort(mod_pc_corr)][:-150]
	# for task in tasks:
	# 	r = nan_pearsonr(df['Diversity Facilitated Modularity Change'][df.Task==task],df['Participation Coefficient'][df.Task==task])
	# 	print task + ': r=' +str(np.around(r[0],3)) +', p=' + str( np.round(r[1],10))
	# with sns.plotting_context("paper",font_scale=1):
	# 	g = sns.FacetGrid(df, col='Task', hue='Task',sharex=True,sharey=True,palette='Paired',size=(6))
	# 	g = g.map(sns.regplot,'Diversity Facilitated Modularity Change','Participation Coefficient',scatter_kws={'alpha':.95})
	# 	plt.tight_layout()
	# 	plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/PCxPCxModularity.pdf',dpi=2400)
	# 	plt.close()		
		"""
		Make a matrix of each node's PC correlation to all edges in the graph.
		"""
		
		predict_nodes = np.arange(len(mod_pc_corr))[np.argsort(mod_pc_corr)][-25:]
		local_predict_nodes = np.arange(len(mod_pc_corr))[np.argsort(mod_pc_corr)][:-150]
		# predict_nodes = np.where(mod_pc_corr>0.0)[0]
		# local_predict_nodes = np.where(mod_pc_corr<0.0)[0]
		pc_thresh = 75
		local_thresh = 25
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		matrices = static_results['matrices']
		pc_edge_corr = pc_edge_correlation(subject_pcs,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr_z.npy' %(project,task,atlas))
		edge_thresh = 75
		edge_thresh = np.percentile(np.nanmean(matrices,axis=0),edge_thresh)
		pc_edge_corr[:,np.nanmean(matrices,axis=0)<edge_thresh] = np.nan
		
		pc_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),pc_thresh)
		local_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),local_thresh)
		connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)>=pc_thresh)[0]
		local_nodes = np.where(np.nanmean(subject_pcs,axis=0)<local_thresh)[0]
		
		pc_edge_corr[:,np.nanmean(matrices,axis=0)<edge_thresh,] = 0.0
		high_pc_edge_matrix = np.nanmean(pc_edge_corr[predict_nodes],axis=0)
		low_pc_edge_matrix = np.nanmean(pc_edge_corr[local_predict_nodes],axis=0)

		# diff_pc_edge_matrix = np.diff([np.nanmean(pc_edge_corr[connector_nodes],axis=0),np.nanmean(pc_edge_corr[local_nodes],axis=0)],axis=0).reshape((264,264))
		matrix = (np.tril(low_pc_edge_matrix) + np.triu(high_pc_edge_matrix)).reshape((264,264))
		# plot_corr_matrix(matrix,network_names.copy(),out_file=None,plot_corr=True,return_array=False)
		#Within and between network edge PC modulation weights in matrix for each network, for each node.
		community_mod_high_wcd = np.zeros((len(predict_nodes),len(np.unique(known_membership))))
		community_mod_low_wcd = np.zeros((len(local_predict_nodes),len(np.unique(known_membership))))
		community_mod_high_bcd = np.zeros((len(predict_nodes),len(np.unique(known_membership))))
		community_mod_low_bcd = np.zeros((len(local_predict_nodes),len(np.unique(known_membership))))
		community_mod_high_wcd_abs = np.zeros((len(predict_nodes),len(np.unique(known_membership))))
		community_mod_low_wcd_abs = np.zeros((len(local_predict_nodes),len(np.unique(known_membership))))
		community_mod_high_bcd_abs = np.zeros((len(predict_nodes),len(np.unique(known_membership))))
		community_mod_low_bcd_abs = np.zeros((len(local_predict_nodes),len(np.unique(known_membership))))
		pc_edge_corr_pos = pc_edge_corr.copy()
		pc_edge_corr_neg = pc_edge_corr.copy()
		pc_edge_corr_abs = np.abs(pc_edge_corr.copy())
		pc_edge_corr_pos[pc_edge_corr_pos<0.0]=0.0
		pc_edge_corr_neg[pc_edge_corr_neg>0.0]=0.0
		
		connector_within_network_mask = pc_edge_corr.copy().astype(bool)
		local_within_network_mask = pc_edge_corr.copy().astype(bool)
		connector_within_network_mask[:,:,:] = False
		local_within_network_mask[:,:,:] = False
		# sm_networks = [0,3,6,8,9,13]
		sm_networks = range(16)
		for n in predict_nodes:
			for community in np.unique(known_membership):
				if community not in sm_networks:
					continue
				community_nodes = np.where(known_membership==community)[0]
				non_community_nodes = np.where(known_membership!=community)[0]
				connector_within_network_mask[n][np.ix_(community_nodes,community_nodes)] = True

		for n in local_predict_nodes:
			for community in np.unique(known_membership):
				if community not in sm_networks:
					continue
				community_nodes = np.where(known_membership==community)[0]
				non_community_nodes = np.where(known_membership!=community)[0]
				local_within_network_mask[n][np.ix_(community_nodes,community_nodes)] = True

		pc_edge_corr_neg = np.abs(pc_edge_corr_neg)
		# print 'all positive'
		# print 'within',scipy.stats.ttest_ind(pc_edge_corr_pos[connector_within_network_mask].reshape(-1),pc_edge_corr_pos[local_within_network_mask].reshape(-1))
		# print 'between', scipy.stats.ttest_ind(pc_edge_corr_pos[connector_within_network_mask==False].reshape(-1),pc_edge_corr_pos[local_within_network_mask==False].reshape(-1))
		# print 'all negative'
		# print 'within', scipy.stats.ttest_ind(pc_edge_corr_neg[connector_within_network_mask].reshape(-1),pc_edge_corr_neg[local_within_network_mask].reshape(-1))
		# print 'between', scipy.stats.ttest_ind(pc_edge_corr_neg[connector_within_network_mask==False].reshape(-1),pc_edge_corr_neg[local_within_network_mask==False].reshape(-1))
		# print 'absolute'
		# print 'within', scipy.stats.ttest_ind(pc_edge_corr_abs[connector_within_network_mask].reshape(-1),pc_edge_corr_abs[local_within_network_mask].reshape(-1))
		# print 'between', scipy.stats.ttest_ind(pc_edge_corr_abs[connector_within_network_mask==False].reshape(-1),pc_edge_corr_abs[local_within_network_mask==False].reshape(-1))

		
		
		def make_strs_for_df(array_to_add,str_to_add):
			array_len = len(array_to_add)
			str_array_ = np.chararray(array_len,itemsize=40)
			str_array_[:] = str_to_add
			return str_array_
		
		def make_array_for_df(arrays_to_add):
			append_array = np.zeros((len(arrays_to_add[0]),len(arrays_to_add))).astype(str)
			append_array[:,0] = arrays_to_add[0]
			append_array[:,1] = arrays_to_add[1]
			append_array[:,2] = arrays_to_add[2]
			return append_array

		violin_columns = ['Changes','Node Type','Edge Type']
		task_violin_df = pd.DataFrame(columns=violin_columns)

		result_array_to_add = pc_edge_corr_pos[connector_within_network_mask].reshape(-1)[pc_edge_corr_pos[connector_within_network_mask].reshape(-1)>0]
		edge_type_ = make_strs_for_df(result_array_to_add,'Within Sub-Network, Positive')
		node_type_ = make_strs_for_df(result_array_to_add,'Connector')
		df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

		result_array_to_add = pc_edge_corr_pos[local_within_network_mask].reshape(-1)[pc_edge_corr_pos[local_within_network_mask].reshape(-1)>0]
		edge_type_ = make_strs_for_df(result_array_to_add,'Within Sub-Network, Positive')
		node_type_ = make_strs_for_df(result_array_to_add,'Local')
		df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)
		
		task_violin_df.Changes = task_violin_df.Changes.astype(float)
		print 'Within Sub-Network, Positive: ' + str(scipy.stats.ttest_ind(task_violin_df.Changes[task_violin_df['Node Type']=='Connector'][task_violin_df['Edge Type']=='Within Sub-Network, Positive'],
			task_violin_df.Changes[task_violin_df['Node Type']=='Local'][task_violin_df['Edge Type']=='Within Sub-Network, Positive']))

		result_array_to_add = pc_edge_corr_pos[connector_within_network_mask==False].reshape(-1)[pc_edge_corr_pos[connector_within_network_mask==False].reshape(-1)>0]
		edge_type_ = make_strs_for_df(result_array_to_add,'Between Sub-Network, Positive')
		node_type_ = make_strs_for_df(result_array_to_add,'Connector')
		df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

		result_array_to_add = pc_edge_corr_pos[local_within_network_mask==False].reshape(-1)[pc_edge_corr_pos[local_within_network_mask==False].reshape(-1)>0]
		edge_type_ = make_strs_for_df(result_array_to_add,'Between Sub-Network, Positive')
		node_type_ = make_strs_for_df(result_array_to_add,'Local')
		df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)
		
		task_violin_df.Changes = task_violin_df.Changes.astype(float)
		print 'Between Sub-Network, Positive: ' + str(scipy.stats.ttest_ind(task_violin_df.Changes[task_violin_df['Node Type']=='Connector'][task_violin_df['Edge Type']=='Between Sub-Network, Positive'],
			task_violin_df.Changes[task_violin_df['Node Type']=='Local'][task_violin_df['Edge Type']=='Between Sub-Network, Positive']))

		result_array_to_add = pc_edge_corr_neg[connector_within_network_mask].reshape(-1)[pc_edge_corr_neg[connector_within_network_mask].reshape(-1)>0]
		edge_type_ = make_strs_for_df(result_array_to_add,'Within Sub-Network, Negative')
		node_type_ = make_strs_for_df(result_array_to_add,'Connector')
		df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

		result_array_to_add = pc_edge_corr_neg[local_within_network_mask].reshape(-1)[pc_edge_corr_neg[local_within_network_mask].reshape(-1)>0]
		edge_type_ = make_strs_for_df(result_array_to_add,'Within Sub-Network, Negative')
		node_type_ = make_strs_for_df(result_array_to_add,'Local')
		df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)
		
		task_violin_df.Changes = task_violin_df.Changes.astype(float)
		print 'Within Sub-Network, Negative: ' + str(scipy.stats.ttest_ind(task_violin_df.Changes[task_violin_df['Node Type']=='Connector'][task_violin_df['Edge Type']=='Within Sub-Network, Negative'],
			task_violin_df.Changes[task_violin_df['Node Type']=='Local'][task_violin_df['Edge Type']=='Within Sub-Network, Negative']))

		result_array_to_add = pc_edge_corr_neg[connector_within_network_mask==False].reshape(-1)[pc_edge_corr_neg[connector_within_network_mask==False].reshape(-1)>0]
		edge_type_ = make_strs_for_df(result_array_to_add,'Between Sub-Network, Negative')
		node_type_ = make_strs_for_df(result_array_to_add,'Connector')
		df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

		result_array_to_add = pc_edge_corr_neg[local_within_network_mask==False].reshape(-1)[pc_edge_corr_neg[local_within_network_mask==False].reshape(-1)>0]
		edge_type_ = make_strs_for_df(result_array_to_add,'Between Sub-Network, Negative')
		node_type_ = make_strs_for_df(result_array_to_add,'Local')
		df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)
		
		task_violin_df.Changes = task_violin_df.Changes.astype(float)
		print 'Between Sub-Network, Negative: ' + str(scipy.stats.ttest_ind(task_violin_df.Changes[task_violin_df['Node Type']=='Connector'][task_violin_df['Edge Type']=='Between Sub-Network, Negative'],
			task_violin_df.Changes[task_violin_df['Node Type']=='Local'][task_violin_df['Edge Type']=='Between Sub-Network, Negative']))

		sns.violinplot(x="Edge Type", y="Changes", hue="Node Type", data=task_violin_df,inner="quart",split=True)
		plt.show()		

		"""
		write values for viz
		"""
			
		# write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		# pcs = np.nanmean(subject_pcs,axis=0)
		# write_df[3] = pcs
		# maxv = np.nanmean(pcs) + (np.nanstd(pcs)*2.5)
		# minv = np.nanmean(pcs) - (np.nanstd(pcs)*2.5)
		# write_df[3][pcs > maxv] = maxv
		# write_df[3][pcs < minv] = minv	
		# write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_%s.node'%(task),sep='\t',index=False,names=False,header=False)
		# write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		# write_df[3] = mod_pc_corr
		# maxv = np.nanmean(mod_pc_corr) + (np.nanstd(mod_pc_corr)*2.5)
		# minv = np.nanmean(mod_pc_corr) - (np.nanstd(mod_pc_corr)*2.5)
		# write_df[3][mod_pc_corr > maxv] = maxv
		# write_df[3][mod_pc_corr < minv] = minv
		# write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_mod_%s.node'%(task),sep='\t',index=False,names=False,header=False)

		#make brains to show the type of analysis we are doing.
		# while True:
		# 	figure_subjects = np.random.randint(0,450,5)
		# 	if (np.argsort(subject_pcs[[figure_subjects],258]) == np.argsort(subject_mods[[figure_subjects]])).all():
		# 		break
		# for subject in figure_subjects:
		# 	write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		# 	write_df[3] = np.nanmean(subject_pcs,axis=0)
		# 	write_df[4] = 3
		# 	write_df[4][258] = 5
		# 	write_df[3] = known_membership
		# 	write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/subject_%s.node'%(subject),sep='\t',index=False,names=False,header=False)
		# 	write_matrix = thresh_matrices[subject].copy()
		# 	for i,j in combinations(range(264),2):
		# 		if i != 258:
		# 			if j != 258:
		# 				write_matrix[i,j] = 0.0
		# 				write_matrix[j,i] = 0.0
		# 	matrix_df = pd.DataFrame(write_matrix)
		# 	matrix_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/subject_%s.edge'%(subject),sep='\t',index=False,names=False,header=False)




	matrix = np.nanmean(pc_edge_corrs,axis=0)
	plot_corr_matrix(matrix,network_names.copy(),out_file='/home/despoB/mb3152/dynamic_mod/figures/avg_pc_edge_corr.pdf',plot_corr=False,return_array=False)

	"""
	difference between a single task and the rest
	"""
	pc_edge_corrs = np.array(pc_edge_corrs)
	network_names.sort()
	for t,i in zip(tasks,range(len(tasks))):
		idx_array = [0,1,2,3,4,5,6]
		idx_array.remove(i)
		plot_corr_matrix(np.diff([np.nanmean(pc_edge_corrs[idx_array],axis=0),pc_edge_corrs[i]],axis=0).reshape(264,264),network_names,
			out_file='/home/despoB/mb3152/dynamic_mod/figures/%s_pc_edge_corr_full_diff_matrix.pdf'%(t),return_array=False)
	"""
	specificity of modulation by nodes' pc?
	check to see if correlation is only there for connector nodes
	also, do absolute valies of the correaltion! 
	is it stronger during task than rest?
	"""
	known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
	known_membership[known_membership==-1] = 0
	atlas = 'power'
	project='hcp'
	for task in tasks:
		pc_thresh = 75
		local_thresh = 25
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas,gamma,omega,msc_cost,window_size)
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		matrices = static_results['matrices']
		pc_edge_corr = pc_edge_correlation(subject_pcs,thresh_matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr.npy' %(project,task,atlas))
		pc_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),pc_thresh)
		connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)>=pc_thresh)[0]
		local_nodes = np.where(np.nanmean(subject_pcs,axis=0)<=local_thresh)[0]
		#sum of weight changes for each node, by each node.
		driver_nodes_list = ['local_nodes','connector_nodes']
		for driver_nodes in driver_nodes_list:
			1/0
			num_nodes = 264
			weight_change_matrix = np.zeros((num_nodes,num_nodes))
			weight_change_matrix_between = np.zeros((num_nodes,num_nodes))
			weight_change_matrix_within = np.zeros((num_nodes,num_nodes))
			weight_change_matrix_between_pos = np.zeros((num_nodes,num_nodes))
			weight_change_matrix_within_pos = np.zeros((num_nodes,num_nodes))
			weight_change_matrix_between_neg = np.zeros((num_nodes,num_nodes))
			weight_change_matrix_within_neg = np.zeros((num_nodes,num_nodes))
			weight_change_matrix_pos = np.zeros((num_nodes,num_nodes))
			weight_change_matrix_neg = np.zeros((num_nodes,num_nodes))
			if driver_nodes == 'local_nodes':
				driver_nodes_array = local_nodes
			else:
				driver_nodes_array = connector_nodes
			for n1,n2 in permutations(range(num_nodes),2):
				if n1 not in driver_nodes_array:
					continue
				mask = np.ones((264),dtype=bool)
				mask[n1] = False
				array = pc_edge_corr[n1][n2][mask]
				weight_change_matrix[n1,n2] = np.nansum(np.abs(array))
				weight_change_matrix_pos[n1,n2] = abs(np.nansum(array[array>0]))
				weight_change_matrix_neg[n1,n2] = abs(np.nansum(array[array<0]))
				# for n3 in range(264):
				# 	if n1 == n3:
				# 		continue
				# 	if known_membership[n3] != known_membership[n2]:
				# 		weight_change_matrix_between[n1,n2] += array[n3]
				# 		if array[n3] > 0.0:
				# 			weight_change_matrix_between_pos[n1,n2] += array[n3]
				# 		else:
				# 			weight_change_matrix_between_neg[n1,n2] += array[n3]
				# 	else:
				# 		weight_change_matrix_within[n1,n2] += array[n3]
				# 		if array[n3] > 0.0:
				# 			weight_change_matrix_within_pos[n1,n2] += array[n3]
				# 		else:
				# 			weight_change_matrix_within_neg[n1,n2] += array[n3]
			print driver_nodes
			temp_matrix = np.nanmean(thresh_matrices,axis=0)
			sns.set_style("white")
			sns.set_style("ticks")
			weight_matrix = weight_change_matrix
			r=pearsonr(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1))
			r = np.round(r[0],3),np.round(r[1],3)
			print pearsonr(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1))
			assert np.max(abs(np.diagonal(weight_matrix))) == 0.0
			with sns.plotting_context("paper",font_scale=1):
				sns.regplot(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1),color='Black',scatter=True,scatter_kws={'alpha':.15},label = r)
				plt.xlabel("Nodes' Diversity Faciliated Absolute Connectivity Changes")
				plt.ylabel('Edge Weight Between Nodes')
				plt.legend(loc='best')
				plt.tight_layout()
				plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/%s_all_connectivity_%s.jpeg'%(task,str(driver_nodes)),dpi=1200)
				# plt.show()
				plt.close()

			temp_matrix = np.nanmean(thresh_matrices,axis=0)
			sns.set_style("white")
			sns.set_style("ticks")
			weight_matrix = weight_change_matrix_pos
			r=pearsonr(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1))
			r = np.round(r[0],3),np.round(r[1],3)
			print pearsonr(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1))
			assert np.max(abs(np.diagonal(weight_matrix))) == 0.0
			with sns.plotting_context("paper",font_scale=1):
				sns.regplot(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1),color='Red',scatter=True,scatter_kws={'alpha':.15},label = r)
				plt.xlabel("Nodes' Diversity Faciliated Positive Connectivity Changes")
				plt.ylabel('Edge Weight Between Nodes')
				plt.legend(loc='best')
				plt.tight_layout()
				plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/%s_pos_connectivity_%s.jpeg'%(task,str(driver_nodes)),dpi=1200)
				# plt.show()
				plt.close()

			temp_matrix = np.nanmean(thresh_matrices,axis=0)
			sns.set_style("white")
			sns.set_style("ticks")
			weight_matrix = weight_change_matrix_neg
			r=pearsonr(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1))
			r = np.round(r[0],3),np.round(r[1],3)
			print pearsonr(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1))
			assert np.max(abs(np.diagonal(weight_matrix))) == 0.0
			with sns.plotting_context("paper",font_scale=1):
				sns.regplot(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1),color='Blue',scatter=True,scatter_kws={'alpha':.15},label = r)
				plt.xlabel("Nodes' Diversity Faciliated Negative Connectivity Changes")
				plt.ylabel('Edge Weight Between Nodes')
				plt.legend(loc='best')
				plt.tight_layout()
				plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/%s_neg_connectivity_%s.jpeg'%(task,str(driver_nodes)),dpi=1200)
				# plt.show()
				plt.close()
	"""
	Are connector nodes modulating the edges that are most variable across subjects?
	"""
	atlas='power'
	gamma=1.0
	omega=0.1
	msc_cost = 0.1
	window_size=100
	network_names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)
	for task in tasks:
		pc_thresh = 75
		local_thresh = 25
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas,gamma,omega,msc_cost,window_size)
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		matrices = static_results['matrices']
		pc_edge_corr = pc_edge_correlation(subject_pcs,thresh_matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr.npy' %(project,task,atlas))
		std_mod = []
		for i in range(264):
			std_mod.append(nan_pearsonr(pc_edge_corr[i].reshape(-1),np.std(matrices,axis=0).reshape(-1))[0])
		print task, pearsonr(np.nanmean(subject_pcs,axis=0),std_mod)
		plot_corr_matrix(np.std(matrices,axis=0),network_names.copy(),out_file=None,plot_corr=True,return_array=False)

def performance_across_tasks(subjects=hcp_subjects):
	tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL']
	project='hcp'
	atlas='power'
	known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
	known_membership[known_membership==-1] = 0
	colors = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[34].values)
	network_names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)
	num_nodes = len(known_membership)
	name_int_dict = {}
	color_int_dict = {}
	for name,color,int_value in zip(network_names,colors,known_membership):
		name_int_dict[int_value] = name
		color_int_dict[int_value] = color
	df = pd.DataFrame(columns=['PC','Task','PCxPerformance','PCxModularity'])
	diff_df = pd.DataFrame(columns=['Task','Modularity_Type','Performance'])
	for task in tasks:
		"""
		see which graph metrics correlate with modularity and performance
		"""
		print task
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		static_results = graph_metrics(subjects,task,atlas)
		# dynamic_results = dynamic_graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs'].copy()
		matrices = static_results['matrices']
		subject_mods = static_results['subject_mods']
		# subject_changes = dynamic_results['subject_changes']
		if task == 'REST':
			task_perf = behavioral_performance(subjects,behavioral_tasks)
			mean_task_perf = np.nanmean(task_perf,axis=1)
		else:
			task_perf = task_performance(subjects,task)
			mean_task_perf = task_perf
		assert subject_pcs.shape[0] == len(subjects)
		mean_pc = np.nanmean(static_results['subject_pcs'],axis=0)
		df_array = []
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
		for node in range(subject_pcs.shape[1]):
			df_array.append([mean_pc[node],task,nan_pearsonr(subject_pcs[:,node],mean_task_perf)[0],mod_pc_corr[node]])
		df = pd.concat([df,pd.DataFrame(df_array,columns=['PC','Task','PCxPerformance','PCxModularity'])],axis=0)

		"""
		predict performance using high and low PCS values. 
		"""
		if task != 'REST':
			to_delete = np.isnan(task_perf).copy()
			to_delete = np.where(to_delete==True)
			subject_pcs = np.delete(subject_pcs,to_delete,axis=0)
			subject_mods = np.delete(subject_mods,to_delete)
			task_perf = np.delete(task_perf,to_delete)
		fit_subject_len = int(len(subjects)*.2)
		pc_thresh = 75
		local_thresh = 25
		pc_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),pc_thresh)
		local_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),local_thresh)
		connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)>=pc_thresh)[0]
		local_nodes = np.where(np.nanmean(subject_pcs,axis=0)<local_thresh)[0]
		subject_pcs[np.isnan(subject_pcs)] = 0.0
		# predict_nodes = connector_nodes
		# local_predict_nodes = local_nodes
		predict_nodes = np.where(mod_pc_corr>0.0)[0]
		local_predict_nodes = np.where(mod_pc_corr<0.0)[0]
		mean_pc = []
		mean_local_pc = []
		for s in range(len(task_perf)):
			mean_pc.append(np.nanmean(scipy.stats.zscore(subject_pcs,axis=1)[s,predict_nodes]))
			mean_local_pc.append(np.nanmean(scipy.stats.zscore(subject_pcs,axis=1)[s,local_predict_nodes]))
			# mean_pc.append(np.nanmean(subject_pcs[s,predict_nodes]))
			# mean_local_pc.append(np.nanmean(subject_pcs[s,local_predict_nodes]))
		diff = np.array(mean_pc)-np.array(mean_local_pc)
		print 't test, median split: ', scipy.stats.ttest_ind(task_perf[np.argsort(diff)[len(diff)/2:]],task_perf[np.argsort(diff)[:len(diff)/2]])
		print 'Correlation between difference of connector and local PC scores: ', nan_pearsonr(scipy.stats.zscore(task_perf),np.array(mean_pc)-np.array(mean_local_pc))
		print 'Correlation between mean PC of Connector Nodes: ', nan_pearsonr(task_perf,np.array(mean_pc))
		print 'Correlation between mean PC of Local Nodes: ', nan_pearsonr(task_perf,np.array(mean_local_pc))
		array_len = len(diff)
		str_list = np.chararray(array_len,itemsize=20)
		str_list[:]= task
		append_array = np.zeros((array_len,3)).astype(str)
		append_array[:,0] = str_list
		append_array[:,2] = scipy.stats.zscore(task_perf)[np.argsort(diff)]
		append_array[:,1] = np.arange((array_len))[np.argsort(diff)].astype(str)
		append_array[:,1][:array_len/2] = 'Integrated'
		append_array[:,1][array_len/2:] = 'Modular'
		diff_df = diff_df.append(pd.DataFrame(data=append_array,columns=['Task','Modularity_Type','Performance']),ignore_index=True)
	# sns.set(style="whitegrid", palette="pastel", color_codes=True)
	# diff_df['Performance'] = diff_df['Performance'].astype(float)
	# sns.violinplot(x="Task", y="Performance", hue="Modularity_Type", data=diff_df,inner="quart",split=True,palette={"Integrated": "b", "Modular": "y"})
	# plt.show()
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=True,sharey=True,palette='Paired',size=(6))
		g = g.map(sns.regplot,'Diversity Facilitated Modularity Change','Participation Coefficient',scatter_kws={'alpha':.95})
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/PCxPCxModularity.pdf',dpi=2400)
		plt.close()	

	"""
	which edges are correlated with task performance?
	"""
	tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL']
	project='hcp'
	atlas='power'
	gamma=1.0
	omega=0.1
	msc_cost = 0.1
	window_size=100
	known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
	known_membership[known_membership==-1] = 0
	colors = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[34].values)
	network_names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)
	num_nodes = len(known_membership)
	name_int_dict = {}
	color_int_dict = {}
	edge_results = []
	results = []
	abs_results = []
	edge_pred_df = pd.DataFrame(columns=['Coefficient','Edge Type'])
	for task in tasks:
		print task
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas,gamma,omega,msc_cost,window_size)
		static_results = graph_metrics(subjects,task,atlas)
		dynamic_results = dynamic_graph_metrics(subjects,task,atlas)
		thresh_matrices = static_results['thresh_matrices']
		task_perf = task_performance(subjects,task)
		to_delete = np.isnan(task_perf).copy()
		to_delete = np.where(to_delete==True)
		task_perf = np.delete(task_perf,to_delete)
		thresh_matrices= np.delete(thresh_matrices,to_delete,axis=0)
		fit_subject_len = int(len(subjects)*.2)
		clf = linear_model.BayesianRidge()
		clf.fit(thresh_matrices.reshape(thresh_matrices.shape[0],-1),task_perf)
		pc_thresh = 75
		local_thresh = 25
		pc_thresh = np.percentile(np.nanmean(static_results['subject_pcs'],axis=0),pc_thresh)
		local_thresh = np.percentile(np.nanmean(static_results['subject_pcs'],axis=0),local_thresh)
		connector_nodes = np.where(np.nanmean(static_results['subject_pcs'],axis=0)>=pc_thresh)[0]
		local_nodes = np.where(np.nanmean(static_results['subject_pcs'],axis=0)<local_thresh)[0]
		edge_results = np.array(clf.coef_).reshape((264,264))
		abs_edge_results = abs(np.array(clf.coef_).reshape((264,264)))
		abs_results.append(abs_edge_results)
		results.append(edge_results)

		to_append = abs_edge_results[np.ix_(connector_nodes,connector_nodes)].reshape(-1)
		str_list = np.chararray(len(to_append),itemsize=20)
		str_list[:]= 'Connector To Connector'
		edge_pred_df = edge_pred_df.append(pd.DataFrame(data=np.array([to_append,str_list]).swapaxes(0,1),columns=['Coefficient','Edge Type']),ignore_index=True)
		to_append = abs_edge_results[np.ix_(connector_nodes)].reshape(-1)
		str_list = np.chararray(len(to_append),itemsize=20)
		str_list[:]= 'All Connector Edges'
		edge_pred_df = edge_pred_df.append(pd.DataFrame(data=np.array([to_append,str_list]).swapaxes(0,1),columns=['Coefficient','Edge Type']),ignore_index=True)
		to_append = abs_edge_results[np.ix_(local_nodes,local_nodes)].reshape(-1)
		str_list = np.chararray(len(to_append),itemsize=20)
		str_list[:]= 'Local To Local'
		edge_pred_df = edge_pred_df.append(pd.DataFrame(data=np.array([to_append,str_list]).swapaxes(0,1),columns=['Coefficient','Edge Type']),ignore_index=True)
		to_append = abs_edge_results[np.ix_(local_nodes)].reshape(-1)
		str_list = np.chararray(len(to_append),itemsize=20)
		str_list[:]= 'All Local Edges'
		edge_pred_df = edge_pred_df.append(pd.DataFrame(data=np.array([to_append,str_list]).swapaxes(0,1),columns=['Coefficient','Edge Type']),ignore_index=True)
	
	subjects = np.array(hcp_subjects).copy()
	subjects = list(subjects)
	subjects = remove_missing_subjects(subjects,'REST',atlas,gamma,omega,msc_cost,window_size)
	rest_static_results = graph_metrics(subjects,'REST',atlas)
	rest_dynamic_results = dynamic_graph_metrics(subjects,'REST')
	
	pc_thresh = 75
	local_thresh = 25
	pc_thresh = np.percentile(np.nanmean(static_results['subject_pcs'],axis=0),pc_thresh)
	local_thresh = np.percentile(np.nanmean(static_results['subject_pcs'],axis=0),local_thresh)
	connector_nodes = np.where(np.nanmean(static_results['subject_pcs'],axis=0)>=pc_thresh)[0]
	local_nodes = np.where(np.nanmean(static_results['subject_pcs'],axis=0)<local_thresh)[0]

	scipy.stats.ttest_ind(mean_results[np.ix_(connector_nodes,connector_nodes)].reshape(-1),mean_results[np.ix_(local_nodes,local_nodes)].reshape(-1))
	scipy.stats.ttest_ind(mean_results[np.ix_(connector_nodes)].reshape(-1),mean_results[np.ix_(local_nodes)].reshape(-1))
	sns.barplot(y='Coefficient',x='Edge Type',data=edge_pred_df,hue='Edge Type', inner="quart")
	plt.show()





	results = []
	predict_nodes = np.concatenate([predict_nodes,local_predict_nodes])
	for i in range(50):
		fit_subjects = np.random.randint(0,len(task_perf),len(task_perf)-fit_subject_len)
		all_subjects = range(len(task_perf))
		test_subjects = []
		for i in all_subjects:
			if i not in fit_subjects:
				test_subjects.append(i)
		clf = linear_model.BayesianRidge()
		clf.fit(subject_pcs[fit_subjects][:,predict_nodes],task_perf[fit_subjects])
		prediction = clf.predict(subject_pcs[test_subjects][:,predict_nodes])
		result = pearsonr(prediction,task_perf[test_subjects])
		# result = np.array([result[0],result[1],np.mean(clf.coef_)])
		results.append(result)
	results = np.array(results)
	mean_r = np.mean(results[:,0])
	mean_p = np.mean(results[:,1])
	sns.set_style("white")
	sns.set_style("ticks")
	print 'connector nodes prediction: ' +  'r=' + str(mean_r) + ' , p=' + str(mean_p)
	with sns.plotting_context("paper",font_scale=1):
		sns.distplot(results[:,0],color=sns.color_palette("coolwarm",7)[-1])
		sns.despine()
		plt.tight_layout()
		plt.title('R values, real performance and predicted performance in %s task' %(task))
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/%s_connector_prediction_hist.jpeg'%(task),dpi=1200)
		plt.close()
	connector_coefficients.append(results[2])
	predict_nodes = local_nodes
	predict_nodes = np.arange(264)[np.argsort(mod_pc_corr)][:35]
	results = []
	for i in range(500):
		fit_subjects = np.random.randint(0,len(task_perf),len(task_perf)-fit_subject_len)
		all_subjects = range(len(task_perf))
		test_subjects = []
		for i in all_subjects:
			if i not in fit_subjects:
				test_subjects.append(i)
		clf = linear_model.BayesianRidge(normalize=True,n_iter=1000)
		clf.fit(subject_pcs[fit_subjects][:,predict_nodes],task_perf[fit_subjects])
		prediction = clf.predict(subject_pcs[test_subjects][:,predict_nodes])
		result = pearsonr(prediction,task_perf[test_subjects])
		result = np.array([result[0],result[1],np.mean(clf.coef_)])
		results.append(result)
		# print 'local nodes prediction: ' +  'r=' + str(result[0]) + ' , p=' + str(result[1])
	results = np.array(results)
	mean_r = np.mean(results[:,0])
	mean_p = np.mean(results[:,1])
	local_coefficients.append(results[2])
	print 'local nodes prediction: ' +  'r=' + str(mean_r) + ' , p=' + str(mean_p)
	sns.set_style("white")
	sns.set_style("ticks")
	with sns.plotting_context("paper",font_scale=1):
		sns.distplot(results[:,0],color=sns.color_palette("coolwarm",7)[0])
		sns.despine()
		plt.tight_layout()
		plt.title('R values, real performance and predicted performance in %s task' %(task))
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/%s_local_prediction_hist.jpeg'%(task),dpi=1200)
		plt.close()

		# def multi_predict(fit_subjects):
		# 	global subject_pcs
		# 	global task_perf
		# 	global predict_nodes
		# 	fit_subjects = np.array(fit_subjects)
		# 	all_subjects = range(len(task_perf))
		# 	clf = linear_model.BayesianRidge(normalize=True)
		# 	test_subjects = []
		# 	for i in all_subjects:
		# 		if i not in fit_subjects:
		# 			test_subjects.append(i)
		# 	clf.fit(subject_pcs[fit_subjects][:,predict_nodes],task_perf[fit_subjects])
		# 	prediction = clf.predict(subject_pcs[test_subjects][:,predict_nodes])
		# 	result = pearsonr(prediction,task_perf[test_subjects])
		# 	result = np.array([result[0],result[1],np.mean(clf.coef_)])
		# 	return result
		
		"""
		connector nodes' prediction
		"""
		
		predict_nodes = connector_nodes
		scores = []
		p_vals = []
		variables = []
		pool = Pool(20)
		# for fit_subjects in combinations(all_subjects,fit_subject_len):
		# 	variables.append([fit_subjects,task_perf,connector_nodes])
		all_subjects = range(len(task_perf))
		results = pool.map(multi_predict,combinations(all_subjects,fit_subject_len))
		for r in result:
			scores.append(result[0])
			p_vals.append(result[1])
			connector_coefficients.append(r[2])
		print 'mean connector prediction: ' + 'r=: ' + str(np.mean(scores)) + ' p=: ' + str(np.mean(p_vals))
		sys.stdout.flush()
		np.save('/home/despoB/mb3152/dynamic_mod/results/connector_pred_%s.npy'%(task),np.array([scores,p_vals,connector_coefficients]))
		"""
		local nodes' prediction
		"""
		# global predict_nodes
		predict_nodes = local_nodes
		scores = []
		p_vals = []
		variables = []
		pool = Pool(20)
		results = pool.map(multi_predict,combinations(all_subjects,fit_subject_len))
		for r in result:
			scores.append(result[0])
			p_vals.append(result[1])
			local_coefficients.append(r[2])
		print 'mean local prediction: ' + 'r=: ' + str(np.mean(scores)) + ' p=: ' + str(np.mean(p_vals))
		sys.stdout.flush()
		np.save('/home/despoB/mb3152/dynamic_mod/results/local_pred_%s.npy'%(task),np.array([scores,p_vals,connector_coefficients]))

	local_coefficients = np.array(local_coefficients).reshape(-1)
	connector_coefficients = np.array(connector_coefficients).reshape(-1)
	all_coefficients = np.array(all_coefficients)

	result = scipy.stats.ttest_ind(np.array(connector_coefficients).reshape(-1),np.array(local_coefficients).reshape(-1))
	print 'average difference in coefficients: ' +'t=' + str(result[0]) + ' , p=' + str(result[1])
	1/0
	coeff_df = []
	for c in connector_coefficients:
		coeff_df.append([c,'Connector Coefficient',' '])
	for l in local_coefficients:
		coeff_df.append([l,'Local Coefficient',' '])
	coeff_df = pd.DataFrame(coeff_df,columns=['Coefficient','Coefficient Type',' '])
	sns.set_style("whitegrid")
	with sns.plotting_context('poster',font_scale=2):
		sns.violinplot(color = 'black',y='Coefficient',linewidth=3,x=' ',data=coeff_df,hue='Coefficient Type',split=True,inner="quartile",palette={'Local Coefficient':sns.color_palette("coolwarm", 7)[0],'Connector Coefficient':sns.color_palette("coolwarm", 7)[-1]})
		plt.show()


	all_coefficients = np.array(all_coefficients)
	write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
	write_df[3] = np.mean(all_coefficients,axis=0)
	write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_coef_mean.node',sep='\t',index=False,names=False,header=False)
	spec_coeff = []
	for c in range(all_coefficients.shape[1]):
		c = all_coefficients[:,c].copy()
		c.sort()
		spec_coeff.append(np.diff([c[-2],c[-1]]))
	write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
	write_df[3] = spec_coeff
	write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_coef_spec.node',sep='\t',index=False,names=False,header=False)

	for task in tasks:
		print ' ' 
		print task + ' Results'
		print 'PCxPerformance, PC'
		print nan_pearsonr(df.PCxPerformance[df.Task==task],df.PC[df.Task==task])
		print 'PCxModularity, PC'
		print nan_pearsonr(df.PCxModularity[df.Task==task],df.PC[df.Task==task])
		print 'PCxModularity, PCxPerformance'
		print nan_pearsonr(df.PCxModularity[df.Task==task],df.PCxPerformance[df.Task==task])
		# print 'Mean PC by Modularity'
		# print pearsonr(subject_mods,np.nanmean(subject_pcs,axis=1))
		# print 'Random'
		# print pearsonr(df.Random_PCxPerformance[df.Task==task],df.PC[df.Task==task])
		# print pearsonr(df.Random_ChangexPerformance[df.Task==task],df.Change[df.Task==task])	
		write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		perf_pc_corr = df.PCxPerformance[df.Task==task]
		write_df[3] = perf_pc_corr
		write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_perf_mod_%s.node'%(task),sep='\t',index=False,names=False,header=False)
	

	#plot those results 
	sns.set_style("white")
	sns.set_style("ticks")
	colors = np.array(sns.palettes.color_palette('Paired',7))
	with sns.plotting_context("paper",font_scale=2):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],size=(6))
		g = g.map(sns.regplot,'PC','PCxPerformance',scatter_kws={'alpha':.95})
		plt.tight_layout()
		plt.show()
	with sns.plotting_context("paper",font_scale=3):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],size=(5))
		g = g.map(sns.regplot,'PCxModularity','PC',scatter_kws={'alpha':.95})
		plt.tight_layout()
		plt.show()
	with sns.plotting_context("paper",font_scale=2):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],size=(6))
		g = g.map(sns.regplot,'PCxModularity','PCxPerformance',scatter_kws={'alpha':.95})
		plt.tight_layout()
		plt.show()


	# print 'PC x PC_Mod Correlation'
	# for i,t in enumerate(tasks):
	# 	print t, nan_pearsonr(mod_pc_corr_array[i],df.PC[df.Task==t])

	# print 'PCxMod and PCxPerformance'
	# for i,t in enumerate(tasks):
	# 	print t, nan_pearsonr(mod_pc_corr_array[i],df.PCxPerformance[df.Task==t])
	
	# print 'Change x Change_Mod Correlation'
	# for i,t in enumerate(tasks):
	# 	print t, nan_pearsonr(mod_change_corr_array[i],df.Change[df.Task==t])
	# mod_pc_corr.append()
	# mod_pc_corr_df = pd.DataFrame(np.array(mod_pc_corr_array).transpose(),columns=[Task,])
	# mod_change_corr_df = pd.DataFrame(np.array(mod_change_corr_array).transpose(),columns=tasks)

def test_norm_edge_weights(subjects=hcp_subjects,atlas='power'):
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
	for task in tasks:	
		for subject in subjects:
			s_matrix = []
			files = glob.glob('/home/despoB/mb3152/dynamic_mod/matrices/%s_%s_*%s*_matrix.npy'%(subject,atlas,task))
			for f in files:
				f = np.load(f)
				np.fill_diagonal(f,0.0)
				f[np.isnan(f)] = 0.0
				f = np.arctanh(f)
				s_matrix.append(f.copy())
			s_matrix = np.nanmean(s_matrix,axis=0)
			
"""
SGE Inputs
"""

if len(sys.argv) > 1:
	if sys.argv[1] == 'perf':
		performance_across_tasks()
	if sys.argv[1] == 'forever':
		a = 0
		while True:
			a = a - 1
			a = a + 1
	if sys.argv[1] == 'pc_edge_corr':
		task = sys.argv[2]
		atlas = 'power'
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		matrices = static_results['matrices']
		pc_edge_corr = pc_edge_correlation(subject_pcs,matrices,path='/home/despoB/mb3152/dynamic_mod/results/hcp_%s_power_pc_edge_corr_z.npy' %(task))
	if sys.argv[1] == 'graph_metrics':
		atlas='power'
		subjects = remove_missing_subjects(list(np.array(hcp_subjects).copy()),sys.argv[2],atlas)
		graph_metrics(subjects,task=sys.argv[2],atlas='power')

"""
Methods

Resting State: ICA FIX plus WB and BP
Tasks: CSF, WM, WB, BP

Results

Figure 1: Analysis Explanation
Single Node's PC across subjects, show Q as well as correlation matrix and edge weights for a single edge.

Figure 2: PC by Diversity Facilitated Modularity Change, all tasks.
Show connectors in red, local in blue (or some other differentiation)

Figure 3: PC by Diversity Facilitated Performance Change, all tasks. 
Show connectors in red, local in blue (or some other differentiation)

Figure 4: Correlation between edge weights and task performance, are these connector node edges or edges modfied by PC changes?
	a.  Correlate edge wights with performance. What types of edges are there? Make map for each task.
	b.	Do correlation between rest edge weights and behavioral measures? Same edges?

Figure 5: Correlation between the stength of an edge (i,j) and how the PC of i facilitates the connectivity changes of j.

Figure 6: Rich Connector Club.
PC results in a higher normalized phi, linear pattern.

"""
