#!/home/despoB/mb3152/anaconda/bin/python
import brain_graphs
import pandas as pd
import os
import sys
import time
import numpy as np
import subprocess
import pickle
import h5py
import random
import time
import scipy
from scipy.io import loadmat
import scipy.io as sio
from scipy.stats.stats import pearsonr
import nibabel as nib
from sklearn.metrics.cluster import normalized_mutual_info_score
from itertools import combinations, permutations
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering
import glob
import math
from collections import Counter
import matplotlib.pylab as plt
plt.rcParams['pdf.fonttype'] = 42
import seaborn as sns
import powerlaw
from richclub import preserve_strength, RC
from multiprocessing import Pool
sys.path.append('/home/despoB/mb3152/dynamic_mod/')
from complexity import FunctionalComplexity_Linear
from sklearn import linear_model, metrics, cross_validation
import random
global hcp_subjects
hcp_subjects = os.listdir('/home/despoB/connectome-data/')
hcp_subjects.sort()
global pc_vals 
global fit_matrices
global task_perf
pc_vals = []
fit_matrices = []
task_perf = []

def nan_pearsonr(x,y):
	x = np.array(x)
	y = np.array(y)
	isnan = np.sum([x,y],axis=0)
	isnan = np.isnan(isnan) == False
	return pearsonr(x[isnan],y[isnan])

def remove_missing_subjects(subjects,task,atlas):
	"""
	remove missing subjects, original array is being edited
	"""
	subjects = list(subjects)
	for subject in subjects:
		files = glob.glob('/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_*%s*_matrix.npy'%(atlas,subject,atlas,task))
		if len(files) < 2:
			subjects.remove(subject)
	return subjects

def print_performance_measure_used(task):
	files = glob.glob('/home/despoB/mb3152/scanner_performance_data/%s_tfMRI_*%s*_Stats.csv' %(100307,task))
	df = pd.read_csv(files[0])
	if task == 'WM':
		print df['ConditionName'][24],df['Measure'][24]
		print df['ConditionName'][27],df['Measure'][27]
		print df['ConditionName'][30],df['Measure'][30]
		print df['ConditionName'][33],df['Measure'][33]
	if task == 'RELATIONAL':
		print df['ConditionName'][0],df['Measure'][0]
		print df['ConditionName'][1],df['Measure'][1]
	if task == 'LANGUAGE':
		print df['ConditionName'][2],df['Measure'][2]
		print df['ConditionName'][5],df['Measure'][5]
	if task == 'SOCIAL':
		print df['ConditionName'][0],df['Measure'][0]
		print df['ConditionName'][5],df['Measure'][5]

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
				if task == 'RELATIONAL':
					t_performance = np.mean([df['Value'][0],df['Value'][1]])
				if task == 'LANGUAGE':
					t_performance = np.mean([df['Value'][2],df['Value'][5]])
					s1 = bdf['ReadEng_AgeAdj'][bdf.Subject == int(subject)] 
					s2 = bdf['PicVocab_AgeAdj'][bdf.Subject == int(subject)]
					t_performance = np.nanmean([t_performance,s1,s2])
				if task == 'SOCIAL':
					t_performance = np.mean([df['Value'][0],df['Value'][5]])
				performance.append(t_performance)
			all_performance.append(np.mean(performance))
		except:
			all_performance.append(np.nan)
	return np.array(all_performance)

def test_reteset_task_performance(subjects,task):
	performance_1 = []
	performance_2 = []
	bdf = pd.read_csv('/home/despoB/mb3152/dynamic_mod/os_behavior_data.csv')	
	for subject in subjects:
		files = glob.glob('/home/despoB/mb3152/scanner_performance_data/%s_tfMRI_*%s*_Stats.csv' %(subject,task))
		try:
			df = pd.read_csv(files[0])
			df = pd.read_csv(files[1])
		except:
			continue
		for i,f in enumerate(files):
			df = pd.read_csv(f)
			if task == 'WM':
				t_performance = np.mean(df['Value'][[24,27,30,33]])
			if task == 'RELATIONAL':
				t_performance = np.mean([df['Value'][0],df['Value'][1]])
			if task == 'LANGUAGE':
				t_performance = np.mean([df['Value'][2],df['Value'][5]])
			if task == 'SOCIAL':
				t_performance = np.mean([df['Value'][0],df['Value'][5]])
			if i == 0:
				performance_1.append(t_performance)
			if i == 1:
				performance_2.append(t_performance)
	print nan_pearsonr(performance_1,performance_2)

def check_motion(subjects):
	for subject in subjects:
	    e = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_12_False_engagement.npy' %(subject))
	    m = np.loadtxt('/home/despoB/mb3152/data/nki_data/preprocessed/pipeline_comp_cor_and_standard/%s_session_1/frame_wise_displacement/_scan_RfMRI_mx_645_rest/FD.1D'%(subject))
	    print pearsonr(m,np.std(e.reshape(900,12),axis=1))

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

def plot_corr_matrix(matrix,membership,out_file=None,block_lower=False,return_array=True,plot_corr=True,label=False):
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
			y_names.append(' ')
			x_names.append(' ')
			y_ticks.append(i)
			x_ticks.append(len(names)-i)
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
	ax.set_yticks(x_ticks)
	ax.set_xticks(y_ticks)
	y_names.reverse()
	std = np.std(corr_mat)
	sns.heatmap(corr_mat,square=True,yticklabels=y_names,xticklabels=x_names,vmin=-3,vmax=3,linewidths=0.0,cmap="RdBu_r")
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

def make_static_matrix(subject,task,project,atlas):
	hcp_subject_dir = '/home/despoB/connectome-data/SUBJECT/*TASK*/*reg*'
	parcel_path = '/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas)
	subject_path = hcp_subject_dir.replace('SUBJECT',subject).replace('TASK',task)
	subject_time_series = brain_graphs.load_subject_time_series(subject_path)
	brain_graphs.time_series_to_matrix(subject_time_series,parcel_path,voxel=False,fisher=False,out_file='/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_%s_matrix.npy' %(subject,atlas,task))

def individual_graph_analyes(variables):
	subject = variables[0]
	print subject
	atlas = variables[1]
	task = variables[2]
	s_matrix = variables[3]
	pc = []
	mod = []
	wmd = []
	for cost in np.array(range(5,16))*0.01:
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

def check_num_nodes(subjects,task,atlas='power'):
	mods = []
	num_nodes = []
	for subject in subjects:
		smods = []
		snum_nodes = []
		print subject
		s_matrix = []
		files = glob.glob('/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_*%s*_matrix.npy'%(atlas,subject,atlas,task))
		for f in files:
			f = np.load(f)
			np.fill_diagonal(f,0.0)
			f[np.isnan(f)] = 0.0
			f = np.arctanh(f)
			s_matrix.append(f.copy())
		if len(s_matrix) == 0:
			continue
		s_matrix = np.nanmean(s_matrix,axis=0)
		for cost in np.array(range(5,16))*0.01:
			temp_matrix = s_matrix.copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True)
			del temp_matrix
			graph = graph.community_infomap(edge_weights='weight')
			smods.append(graph.modularity)
			snum_nodes.append(len(np.array(graph.graph.degree())[np.array(graph.graph.degree())>0.0]))
		mods.append(np.mean(smods))
		num_nodes.append(np.mean(snum_nodes))
		print pearsonr(mods,num_nodes)

def check_normalize(subjects,task,atlas='power'):	
	for subject in subjects:
		print subject
		s_matrix = []
		files = glob.glob('/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_*%s*_matrix.npy'%(atlas,subject,atlas,task))
		for f in files:
			f = np.load(f)
			np.fill_diagonal(f,0.0)
			f[np.isnan(f)] = 0.0
			f = np.arctanh(f)
			s_matrix.append(f.copy())
		if len(s_matrix) == 0:
			continue
		s_matrix = np.nanmean(s_matrix,axis=0)
		for cost in np.array(range(5,16))*0.01:
			temp_matrix = s_matrix.copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True)
			assert np.diff([cost,graph.density()])[0] < .005
			assert np.min(graph.get_adjacency(attribute='weight').data) >= 0.0

def graph_metrics(subjects,task,atlas,project='hcp',run_version='fz',run=False):
	"""
	run graph metrics or load them
	"""
	if run == False:
		done_subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_%s.npy' %(project,task,atlas,run_version)) 
		assert (done_subjects == subjects).all() #make sure you are getting subjects / subjects order you wanted and ran last time.
		subject_pcs = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pcs_%s.npy' %(project,task,atlas,run_version)) 
		subject_wmds = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_wmds_%s.npy' %(project,task,atlas,run_version)) 
		subject_mods = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_mods_%s.npy' %(project,task,atlas,run_version)) 
		matrices = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_matrices_%s.npy' %(project,task,atlas,run_version)) 
		thresh_matrices = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_z_matrices_%s.npy' %(project,task,atlas,run_version))
	elif run == True:
		variables = []
		matrices = []
		thresh_matrices = []
		for subject in subjects:
			s_matrix = []
			files = glob.glob('/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_*%s*_matrix.npy'%(atlas,subject,atlas,task))
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
		pool = Pool(16)
		results = pool.map(individual_graph_analyes,variables)		
		for r,s in zip(results,subjects):
			subject_mods.append(np.nanmean(r[0]))
			subject_pcs.append(r[1])
			subject_wmds.append(r[2])
			assert r[3] == s #make sure it returned the order of subjects/results correctly
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pcs_%s.npy' %(project,task,atlas,run_version),np.array(subject_pcs))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_wmds_%s.npy' %(project,task,atlas,run_version),np.array(subject_wmds))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_mods_%s.npy' %(project,task,atlas,run_version),np.array(subject_mods))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_%s.npy' %(project,task,atlas,run_version),np.array(subjects))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_matrices_%s.npy'%(project,task,atlas,run_version),np.array(matrices))
		np.save('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_z_matrices_%s.npy'%(project,task,atlas,run_version),np.array(thresh_matrices))
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

def network_labels(atlas):
	if atlas == 'gordon':
		name_dict = {}
		df = pd.read_excel('/home/despoB/mb3152/dynamic_mod/Parcels.xlsx')
		df.Community[df.Community=='None'] = 'Uncertain'
		for i,com in enumerate(np.unique(df.Community.values)):
			name_dict[com] = i
		known_membership = np.zeros((333))
		for i in range(333):
			known_membership[i] = name_dict[df.Community[i]]
		network_names = np.array(df.Community.values).astype(str)
	if atlas == 'power':
		known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
		known_membership[known_membership==-1] = 0
		network_names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)
	name_int_dict = {}
	for name,int_value in zip(network_names,known_membership):
		name_int_dict[int_value] = name
	return known_membership,network_names,len(known_membership),name_int_dict

def split_connectivity_across_tasks(n_iters=10000):
	global hcp_subjects
	try:
		df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/split_corrs.csv')
	except:
		split = True
		tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
		project='hcp'
		atlas = 'power'
		known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
		df_columns=['Task','Pearson R, PC, PC & Q','Pearson R, WCD, WCD & Q']
		df = pd.DataFrame(columns = df_columns)
		for task in tasks:
			print task
			subjects = np.array(hcp_subjects).copy()
			subjects = list(subjects)
			subjects = remove_missing_subjects(subjects,task,atlas)
			assert (subjects == np.load('/home/despoB/mb3152/dynamic_mod/results/hcp_%s_%s_subs_fz.npy'%(task,atlas))).all()
			static_results = graph_metrics(subjects,task,atlas)
			subject_pcs = static_results['subject_pcs']
			subject_mods = static_results['subject_mods']
			subject_wmds = static_results['subject_wmds']
			matrices = static_results['matrices']
			task_perf = task_performance(subjects,task)
			assert subject_pcs.shape[0] == len(subjects)
			wmd_rs = []
			pc_rs = []
			from sklearn.cross_validation import ShuffleSplit
			for pc_subs,pc_mod_subjects in ShuffleSplit(n=len(subjects),n_iter=n_iters,train_size=.5,test_size=.5):
				mod_pc_corr = np.zeros(subject_pcs.shape[1])
				mod_wmd_corr = np.zeros(subject_pcs.shape[1])
				mean_pc = np.nanmean(subject_pcs[pc_subs,],axis=0)
				mean_wmd = np.nanmean(subject_wmds[pc_subs,],axis=0)
				for i in range(subject_pcs.shape[1]):
					mod_pc_corr[i] = nan_pearsonr(subject_mods[pc_mod_subjects],subject_pcs[pc_mod_subjects,i])[0]
					mod_wmd_corr[i] = nan_pearsonr(subject_mods[pc_mod_subjects],subject_wmds[pc_mod_subjects,i])[0]
				df = df.append({'Task':task,'Pearson R, PC, PC & Q':nan_pearsonr(mod_pc_corr,mean_pc)[0],'Pearson R, WCD, WCD & Q':nan_pearsonr(mod_wmd_corr,mean_wmd)[0]},ignore_index=True)
			print np.mean(df['Pearson R, PC, PC & Q'][df.Task==task])
			print np.mean(df['Pearson R, WCD, WCD & Q'][df.Task==task])
		df.to_csv('/home/despoB/mb3152/dynamic_mod/results/split_corrs.csv')
	sns.plt.figure(figsize=(20,10))
	sns.violinplot(data=df,y='Pearson R, PC, PC & Q',x='Task',inner='quartile')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/split_pc.pdf',dpi=3600)
	sns.plt.close()
	sns.violinplot(data=df,y='Pearson R, WCD, WCD & Q',x='Task',inner='quartile')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/split_wcd.pdf',dpi=3600)
	sns.plt.close()

def connectivity_across_tasks(atlas='power',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']):
	global hcp_subjects
	split = True
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
	project='hcp'
	atlas = 'power'
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	df_columns=['PC','WCD','Task','Pearson R, PC & Q','Pearson R, WCD & Q','PC-Q Coefficients','WMD-Q Coefficients','PC_Q+','WCD_Q+']
	df = pd.DataFrame(columns = df_columns)
	loo_columns=['Task','Predicted Q','Q']
	loo_df = pd.DataFrame(columns = loo_columns)
	for task in tasks:
		print task
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		assert (subjects == np.load('/home/despoB/mb3152/dynamic_mod/results/hcp_%s_%s_subs_fz.npy'%(task,atlas))).all()
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		subject_mods = static_results['subject_mods']
		subject_wmds = static_results['subject_wmds']
		matrices = static_results['matrices']
		task_perf = task_performance(subjects,task)
		assert subject_pcs.shape[0] == len(subjects)
		mean_pc = np.nanmean(subject_pcs,axis=0)
		mean_wmd = np.nanmean(subject_wmds,axis=0)
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
		mod_wmd_corr = np.zeros(subject_wmds.shape[1])
		for i in range(subject_wmds.shape[1]):
			mod_wmd_corr[i] = nan_pearsonr(subject_mods,subject_wmds[:,i])[0]
		print 'Pearson R, PC & Q, Mean PC: ', nan_pearsonr(mod_pc_corr,mean_pc)
		print 'Pearson R, PC & WCD, Mean WMD: ', nan_pearsonr(mod_wmd_corr,mean_wmd)
		assert (mod_pc_corr==0).all() == False
		assert (mod_wmd_corr==0).all() == False
		high_pc_mod = []
		high_wcd_mod = []
		for s in range(subject_pcs.shape[0]):
			high_pc_mod.append(np.nanmean(subject_pcs[s][np.where(mod_pc_corr>0)])-np.nanmean(subject_pcs[s][np.where(mod_pc_corr<0)]))
			high_wcd_mod.append(np.nanmean(subject_wmds[s][np.where(mod_wmd_corr>0)])-np.nanmean(subject_wmds[s][np.where(mod_wmd_corr<0)]))
			# high_pc_mod.append(np.nanmean(subject_pcs[s][np.where(mod_pc_corr>0)]))
			# high_wcd_mod.append(np.nanmean(subject_wmds[s][np.where(mod_wmd_corr>0)]))
		print 'Pearson R, PC Q+, WCD Q+: ', pearsonr(high_pc_mod,high_wcd_mod)

		pc_vals = subject_pcs.copy()
		wmd_vals = subject_wmds.copy()
		pvals = np.concatenate([wmd_vals,pc_vals],axis=1)
		pvals[np.isnan(pvals)] = 0.0

		clf = linear_model.BayesianRidge()
		clf.fit(pvals,subject_mods)
		print 'Nodal Role Prediction of Q: ', pearsonr(clf.predict(pvals),subject_mods)
		print 'WMD, Coefficients of Q', pearsonr(mean_pc,clf.coef_[264:])
		print 'PC, Coefficients of Q', pearsonr(mean_wmd,clf.coef_[:264])
		df_array = []
		for node in range(num_nodes):
			df_array.append([mean_pc[node],mean_wmd[node],task,mod_pc_corr[node],mod_wmd_corr[node],clf.coef_[264+node],clf.coef_[node],high_pc_mod[node],high_wcd_mod[node]])
		df = pd.concat([df,pd.DataFrame(df_array,columns=df_columns)],axis=0)

		# vs = []
		# for t in range(pvals.shape[0]):
		# 	train = np.ones(len(pvals)).astype(bool)
		# 	train[t] = False
		# 	vs.append([pvals,train,t,subject_mods])
		# pool = Pool(20)
		# q_prediction = pool.map(predict,vs)		
		# print 'Nodal Role Prediction of Q, LOO: ', pearsonr(q_prediction,subject_mods)
		# loo_array = []
		# for i in range(len(q_prediction)):
		# 	loo_array.append([task,q_prediction[i],subject_mods[i]])
		# loo_df = pd.concat([loo_df,pd.DataFrame(loo_array,columns=loo_columns)],axis=0)
		# # make brain figures
		import matlab
		eng = matlab.engine.start_matlab()
		eng.addpath('/home/despoB/mb3152/BrainNet/')
		# pc values
		write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		pcs = np.nanmean(subject_pcs,axis=0)
		write_df[3] = pcs
		write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_%s.node'%(task),sep='\t',index=False,names=False,header=False)
		node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_%s.node'%(task)
		surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
		img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/pc_%s.png' %(task)
		configs = '/home/despoB/mb3152/BrainNet/pc_values.mat'
		eng.BrainNet_MapCfg(node_file,surf_file,configs,img_file)
		#mod pc values
		write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		write_df[3] = mod_pc_corr
		maxv = np.nanmean(mod_pc_corr) + (np.nanstd(mod_pc_corr)*2.75)
		minv = np.nanmean(mod_pc_corr) - (np.nanstd(mod_pc_corr)*2.75)
		write_df[3][mod_pc_corr > maxv] = maxv
		write_df[3][mod_pc_corr < minv] = minv
		write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_mod_%s.node'%(task),sep='\t',index=False,names=False,header=False)			
		node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_mod_%s.node'%(task)
		surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
		img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/mod_pc_corr_%s.png' %(task)
		configs = '/home/despoB/mb3152/BrainNet/pc_values.mat'
		eng.BrainNet_MapCfg(node_file,surf_file,configs,img_file)
		# wcd values
		write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		wmds = np.nanmean(subject_wmds,axis=0)
		write_df[3] = wmds
		write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_wmds_%s.node'%(task),sep='\t',index=False,names=False,header=False)
		node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/power_wmds_%s.node'%(task)
		surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
		img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/wmds_%s.png' %(task)
		configs = '/home/despoB/mb3152/BrainNet/pc_values.mat'
		eng.BrainNet_MapCfg(node_file,surf_file,configs,img_file)
		#mod wcd values
		write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		write_df[3] = mod_wmd_corr
		maxv = np.nanmean(mod_wmd_corr) + (np.nanstd(mod_wmd_corr)*2.75)
		minv = np.nanmean(mod_wmd_corr) - (np.nanstd(mod_wmd_corr)*2.75)
		write_df[3][mod_wmd_corr > maxv] = maxv
		write_df[3][mod_wmd_corr < minv] = minv
		write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_wmd_mod_%s.node'%(task),sep='\t',index=False,names=False,header=False)			
		node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/power_wmd_mod_%s.node'%(task)
		surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
		img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/mod_wmd_corr_%s.png' %(task)
		configs = '/home/despoB/mb3152/BrainNet/pc_values.mat'
		eng.BrainNet_MapCfg(node_file,surf_file,configs,img_file)

	if split == True:
		plot_connectivity_results(df,'PC','Pearson R, PC & Q','/home/despoB/mb3152/dynamic_mod/figures/PCxPCxModularity_split.pdf')
		plot_connectivity_results(df,'WCD','Pearson R, WCD & Q','/home/despoB/mb3152/dynamic_mod/figures/WCDxWCDxModularity_split.pdf')
		1/0
	plot_connectivity_results(df,'PC_Q+','WCD_Q+','/home/despoB/mb3152/dynamic_mod/figures/PC_QplusxWCD_Qplus.pdf')
	plot_connectivity_results(loo_df,'Predicted Q','Q','/home/despoB/mb3152/dynamic_mod/figures/WMD_PC_Q_Prediction.pdf')
	plot_connectivity_results(df,'PC','PC-Q Coefficients','/home/despoB/mb3152/dynamic_mod/figures/PC-Q_Coefs.pdf')
	plot_connectivity_results(df,'WCD','WMD-Q Coefficients','/home/despoB/mb3152/dynamic_mod/figures/WMD-Q_Coefs.pdf')

	plot_connectivity_results(df,'PC','Pearson R, PC & Q','/home/despoB/mb3152/dynamic_mod/figures/PCxPCxModularity.pdf')
	plot_connectivity_results(df,'WCD','Pearson R, WCD & Q','/home/despoB/mb3152/dynamic_mod/figures/WCDxWCDxModularity.pdf')
	plot_connectivity_results(df,'WCD','PC','/home/despoB/mb3152/dynamic_mod/figures/WMDxPC.pdf')

def matrix_of_changes():
	"""
	Make a matrix of each node's PC correlation to all edges in the graph.
	"""
	drivers = ['PC','WCD']
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
	project='hcp'
	atlas = 'power'
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	for driver in drivers:
		print driver
		all_matrices = []
		violin_df = pd.DataFrame()
		for task in tasks:
			print task
			subjects = np.array(hcp_subjects).copy()
			subjects = list(subjects)
			subjects = remove_missing_subjects(subjects,task,atlas)
			assert (subjects == np.load('/home/despoB/mb3152/dynamic_mod/results/hcp_%s_%s_subs_fz.npy'%(task,atlas))).all()
			static_results = graph_metrics(subjects,task,atlas)
			subject_pcs = static_results['subject_pcs']
			subject_mods = static_results['subject_mods']
			subject_wmds = static_results['subject_wmds']
			matrices = static_results['matrices']
			task_perf = task_performance(subjects,task)
			assert subject_pcs.shape[0] == len(subjects)
			mean_pc = np.nanmean(subject_pcs,axis=0)
			mean_wmd = np.nanmean(subject_wmds,axis=0)
			mod_pc_corr = np.zeros(subject_pcs.shape[1])
			for i in range(subject_pcs.shape[1]):
				mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
			mod_wmd_corr = np.zeros(subject_wmds.shape[1])
			for i in range(subject_wmds.shape[1]):
				mod_wmd_corr[i] = nan_pearsonr(subject_mods,subject_wmds[:,i])[0]
			if driver == 'pc':
				predict_nodes = np.where(mod_pc_corr>0.0)[0]
				local_predict_nodes = np.where(mod_pc_corr<0.0)[0]
				pc_edge_corr = np.arctanh(pc_edge_correlation(subject_pcs,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr_z.npy' %(project,task,atlas)))
			else:		
				predict_nodes = np.where(mod_wmd_corr>0.0)[0]
				local_predict_nodes = np.where(mod_wmd_corr<0.0)[0]
				pc_edge_corr = np.arctanh(pc_edge_correlation(subject_wmds,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_wmd_edge_corr_z.npy' %(project,task,atlas)))
			# Plot matrix of changes
			edge_thresh = 75
			edge_thresh = np.percentile(np.nanmean(matrices,axis=0),edge_thresh)
			pc_edge_corr[:,np.nanmean(matrices,axis=0)<edge_thresh,] = np.nan
			high_pc_edge_matrix = np.nanmean(pc_edge_corr[predict_nodes],axis=0)
			low_pc_edge_matrix = np.nanmean(pc_edge_corr[local_predict_nodes],axis=0)
			matrix = (np.tril(low_pc_edge_matrix) + np.triu(high_pc_edge_matrix)).reshape((264,264))
			plot_matrix = matrix.copy()
			plot_matrix_mask = np.isnan(plot_matrix)
			zscores = scipy.stats.zscore(plot_matrix[plot_matrix_mask==False].reshape(-1))
			plot_matrix[plot_matrix_mask==False] = zscores
			if task != 'REST':
				all_matrices.append(plot_matrix)
			plot_corr_matrix(plot_matrix,network_names.copy(),out_file='/home/despoB/mb3152/dynamic_mod/figures/%s_corr_matrix_%s.pdf'%(driver,task),plot_corr=False,return_array=False)

			pc_edge_corr[np.isnan(pc_edge_corr)] = 0.0
			connector_within_network_mask = pc_edge_corr.copy().astype(bool)
			local_within_network_mask = pc_edge_corr.copy().astype(bool)
			connector_between_network_mask = pc_edge_corr.copy().astype(bool)
			local_between_network_mask = pc_edge_corr.copy().astype(bool)
			connector_within_network_mask[:,:,:] = False
			local_within_network_mask[:,:,:] = False
			connector_between_network_mask[:,:,:] = False
			local_between_network_mask[:,:,:] = False
			
			for n in predict_nodes:
				for node1,node2 in combinations(range(264),2):
					if n == node1:
						continue
					if n == node2:
						continue
					if known_membership[node1] == 0:
						continue
					if known_membership[node2] == 0:
						continue
					if known_membership[node1] == known_membership[node2]:
						connector_within_network_mask[n][node1,node2] = True
						connector_within_network_mask[n][node2,node1] = True
					else:
						connector_between_network_mask[n][node1,node2] = True
						connector_between_network_mask[n][node2,node1] = True

			for n in local_predict_nodes:
				for node1,node2 in combinations(range(264),2):
					if n == node1:
						continue
					if n == node2:
						continue
					if known_membership[node1] == 0:
						continue
					if known_membership[node2] == 0:
						continue
					if known_membership[node1] == known_membership[node2]:
						local_within_network_mask[n][node1,node2] = True
						local_within_network_mask[n][node2,node1] = True
					else:
						local_between_network_mask[n][node1,node2] = True
						local_between_network_mask[n][node2,node1] = True

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

			violin_columns = ["r value, node i's PCs and j's edge weights","Node Type","Edge Type"]
			task_violin_df = pd.DataFrame(columns=violin_columns)
			result_array_to_add = pc_edge_corr[connector_within_network_mask].reshape(-1)[pc_edge_corr[connector_within_network_mask].reshape(-1)!=0.0]
			edge_type_ = make_strs_for_df(result_array_to_add,'Within Community')
			node_type_ = make_strs_for_df(result_array_to_add,'Q+')
			df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
			task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

			result_array_to_add = pc_edge_corr[local_within_network_mask].reshape(-1)[pc_edge_corr[local_within_network_mask].reshape(-1)!=0.0]
			edge_type_ = make_strs_for_df(result_array_to_add,'Within Community')
			node_type_ = make_strs_for_df(result_array_to_add,'Q-')
			df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
			task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

			result_array_to_add = pc_edge_corr[connector_between_network_mask].reshape(-1)[pc_edge_corr[connector_between_network_mask].reshape(-1)!=0.0]
			edge_type_ = make_strs_for_df(result_array_to_add,'Between Community')
			node_type_ = make_strs_for_df(result_array_to_add,'Q+')
			df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
			task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

			result_array_to_add = pc_edge_corr[local_between_network_mask].reshape(-1)[pc_edge_corr[local_between_network_mask].reshape(-1)!=0.0]
			edge_type_ = make_strs_for_df(result_array_to_add,'Between Community')
			node_type_ = make_strs_for_df(result_array_to_add,'Q-')
			df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
			task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)
			task_violin_df["r value, node i's PCs and j's edge weights"] = task_violin_df["r value, node i's PCs and j's edge weights"].astype(float)

			print 'Within V Between, Connectors: ' + str(scipy.stats.ttest_ind(task_violin_df["r value, node i's PCs and j's edge weights"][task_violin_df['Node Type']=='Q+'][task_violin_df['Edge Type']=='Within Community'],
				task_violin_df["r value, node i's PCs and j's edge weights"][task_violin_df['Node Type']=='Q+'][task_violin_df['Edge Type']=='Between Community']))
			print 'Within V Between, Local: ' + str(scipy.stats.ttest_ind(task_violin_df["r value, node i's PCs and j's edge weights"][task_violin_df['Node Type']=='Q-'][task_violin_df['Edge Type']=='Within Community'],
				task_violin_df["r value, node i's PCs and j's edge weights"][task_violin_df['Node Type']=='Q-'][task_violin_df['Edge Type']=='Between Community']))
			#append for average of all
			violin_df = violin_df.append(pd.DataFrame(data=task_violin_df,columns=violin_columns),ignore_index=True)
			#Figure for single Task
			sns.set_style("white")
			sns.set_style("ticks")
			colors = sns.color_palette(['#fdfd96','#C4D8E2'])
			with sns.plotting_context("paper",font_scale=2):
				plt.figure(figsize=(24,16))
				sns.boxplot(x="Node Type", y="r value, node i's PCs and j's edge weights", hue="Edge Type", order=['Q+','Q-'], data=task_violin_df)
				plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/%s_edge_mod_%s.pdf'%(driver,task),dpi=4600)
				plt.close()
		# Average of All
		plot_corr_matrix(np.nanmean(all_matrices,axis=0),network_names.copy(),out_file='/home/despoB/mb3152/dynamic_mod/figures/%s_corr_matrix_avg.pdf'%(driver),plot_corr=False,return_array=False)
		print 'Within V Between, Connectors: ' + str(scipy.stats.ttest_ind(violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q+'][violin_df['Edge Type']=='Within Community'],
			violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q+'][violin_df['Edge Type']=='Between Community']))
		print 'Within V Between, Local: ' + str(scipy.stats.ttest_ind(violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q-'][violin_df['Edge Type']=='Within Community'],
			violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q-'][violin_df['Edge Type']=='Between Community']))
		sns.set_style("white")
		sns.set_style("ticks")
		colors = sns.color_palette(['#fdfd96','#C4D8E2'])
		with sns.plotting_context("paper",font_scale=3):
			plt.figure(figsize=(24,16))
			sns.boxplot(x="Node Type", y="r value, node i's PCs and j's edge weights",hue="Edge Type", palette=colors,order=['Q+','Q-'], data=violin_df)
			plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/%s_edge_mod_avg.pdf'%(driver),dpi=4600)
			plt.close()

def specificity():
	"""
	Specificity of modulation by nodes' PC.
	Does the PC value of i impact the connectivity of j as i and j are more strongly connected?
	"""
	atlas = 'power'
	project='hcp'
	df_columns=['Task','Hub Measure','Q+/Q-','Average Edge i-j Weight',"Strength of r's, i's PC & j's Q"]
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	df = pd.DataFrame(columns = df_columns)
	for task in tasks:
		print task
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		subject_wmds = static_results['subject_wmds']
		subject_mods = static_results['subject_mods']
		subject_wmds = static_results['subject_wmds']
		matrices = static_results['matrices']
		#sum of weight changes for each node, by each node.
		hub_nodes = ['PC','WMD']
		driver_nodes_list = ['Q+','Q-']
		mean_pc = np.nanmean(subject_pcs,axis=0)
		mean_wmd = np.nanmean(subject_wmds,axis=0)
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
		mod_wmd_corr = np.zeros(subject_wmds.shape[1])
		for i in range(subject_wmds.shape[1]):
			mod_wmd_corr[i] = nan_pearsonr(subject_mods,subject_wmds[:,i])[0]
		for hub_node in hub_nodes:
			print hub_node
			if hub_node == 'PC':
				pc_edge_corr = np.arctanh(pc_edge_correlation(subject_pcs,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr_z.npy' %(project,task,atlas)))
				connector_nodes = np.where(mod_pc_corr>0.0)[0]
				local_nodes = np.where(mod_pc_corr<0.0)[0]
			else:
				pc_edge_corr = np.arctanh(pc_edge_correlation(subject_wmds,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_wmd_edge_corr_z.npy' %(project,task,atlas)))
				connector_nodes = np.where(mod_wmd_corr>0.0)[0]
				local_nodes = np.where(mod_wmd_corr<0.0)[0]
			edge_thresh_val = 75.0
			edge_thresh = np.percentile(np.nanmean(matrices,axis=0),edge_thresh_val)
			pc_edge_corr[:,np.nanmean(matrices,axis=0)<edge_thresh] = np.nan
			for driver_nodes in driver_nodes_list:
				weight_change_matrix = np.zeros((num_nodes,num_nodes))
				weight_change_matrix_between = np.zeros((num_nodes,num_nodes))
				weight_change_matrix_within = np.zeros((num_nodes,num_nodes))
				if driver_nodes == 'Q-':
					driver_nodes_array = local_nodes
				else:
					driver_nodes_array = connector_nodes
				for n1,n2 in permutations(range(num_nodes),2):
					if n1 not in driver_nodes_array:
						continue
					if known_membership[n2] == 0:
						continue
					mask = np.ones((num_nodes),dtype=bool)
					mask[n1] = False
					array = pc_edge_corr[n1][n2]
					masked_array = array[mask]
					weight_change_matrix[n1,n2] = np.nansum(np.abs(masked_array))
					for n3 in range(264):
						if n1 == n3:
							continue
						if known_membership[n3]!= known_membership[n2]:
							weight_change_matrix_between[n1,n2] = np.nansum([weight_change_matrix_between[n1,n2],array[n3]])
						else:
							weight_change_matrix_within[n1,n2] = np.nansum([weight_change_matrix_within[n1,n2],array[n3]])
				temp_matrix = np.nanmean(matrices,axis=0)
				weight_matrix = weight_change_matrix_within-weight_change_matrix_between
				weight_matrix[np.isnan(weight_matrix)] = 0.0
				df_columns=['Task','Hub Measure','Q+/Q-','Average Edge i-j Weight',"Strength of r's, i's PC & j's Q"]
				df_array = []
				for i,j in zip(temp_matrix[weight_matrix!=0.0].reshape(-1),weight_matrix[weight_matrix!=0.0].reshape(-1)):
					df_array.append([task,hub_node,driver_nodes,i,j])
				df = pd.concat([df,pd.DataFrame(df_array,columns=df_columns)],axis=0)
				print driver_nodes
				print pearsonr(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1))

	plot_connectivity_results(df[(df['Q+/Q-']=='Q+') &(df['Hub Measure']=='PC')],"Strength of r's, i's PC & j's Q",'Average Edge i-j Weight','/home/despoB/mb3152/dynamic_mod/figures/edge_spec_pcqplus_%s.pdf'%(edge_thresh_val))
	plot_connectivity_results(df[(df['Q+/Q-']=='Q-') &(df['Hub Measure']=='PC')],"Strength of r's, i's PC & j's Q",'Average Edge i-j Weight','/home/despoB/mb3152/dynamic_mod/figures/edge_spec_pcqminus_%s.pdf'%(edge_thresh_val))
	plot_connectivity_results(df[(df['Q+/Q-']=='Q+') &(df['Hub Measure']=='WMD')],"Strength of r's, i's PC & j's Q",'Average Edge i-j Weight','/home/despoB/mb3152/dynamic_mod/figures/edge_spec_wmdqplus_%s.pdf'%(edge_thresh_val))
	plot_connectivity_results(df[(df['Q+/Q-']=='Q-') &(df['Hub Measure']=='WMD')],"Strength of r's, i's PC & j's Q",'Average Edge i-j Weight','/home/despoB/mb3152/dynamic_mod/figures/edge_spec_wmdqminus_%s.pdf'%(edge_thresh_val))
	# """
	# Are connector nodes modulating the edges that are most variable across subjects?
	# """
	# atlas='power'
	# known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	# for task in tasks:
	# 	pc_thresh = 75
	# 	local_thresh = 25
	# 	subjects = np.array(hcp_subjects).copy()
	# 	subjects = list(subjects)
	# 	subjects = remove_missing_subjects(subjects,task,atlas)
	# 	static_results = graph_metrics(subjects,task,atlas)
	# 	subject_pcs = static_results['subject_pcs']
	# 	subject_wmds = static_results['subject_wmds']
	# 	matrices = static_results['matrices']
	# 	matrices[:,np.nanmean(matrices,axis=0)<0.0] = np.nan
	# 	pc_edge_corr = np.arctanh(pc_edge_correlation(subject_wmds,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_wmd_edge_corr_z.npy' %(project,task,atlas)))
	# 	# pc_edge_corr = pc_edge_correlation(subject_pcs,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr_z.npy' %(project,task,atlas))
	# 	std_mod = []
	# 	tstd = np.std(matrices,axis=0).reshape(-1)
	# 	for i in range(num_nodes):
	# 		std_mod.append(nan_pearsonr(pc_edge_corr[i].reshape(-1),tstd)[0])
	# 	# print task, pearsonr(np.nanmean(subject_pcs,axis=0),std_mod)
	# 	print task, pearsonr(np.nanmean(subject_wmds,axis=0),std_mod)
	# 	plot_corr_matrix(np.std(matrices,axis=0),network_names.copy(),out_file=None,plot_corr=True,return_array=False)

def between_community_centrality(graph):
	vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	pc = vc.pc
	pc[np.isnan(pc)] = 0.0
	deg = np.array(vc.community.graph.strength(weights='weight'))
	return [np.array(graph.betweenness(weights='weight'))[np.argsort(pc)[-25:]],
	np.array(graph.betweenness(weights='weight'))[np.argsort(deg)[-25:]]]

def attack(graph):
	vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	pc = vc.pc
	pc[np.isnan(pc)] = 0.0
	deg = np.array(vc.community.graph.strength(weights='weight'))
	cmask = brain_graphs.community_matrix(vc.community.membership,0).astype(bool)
	nidx = graph.vcount()/5
	connector_nodes = np.argsort(pc)[-nidx:]
	degree_nodes = np.argsort(deg)[-nidx:]
	healthy_sp = np.sum(np.array(graph.shortest_paths()))
	degree_edges = []
	for d in combinations(degree_nodes,2):
		if graph[d[0],d[1]] > 0:
			degree_edges.append([d])
	connector_edges = []
	for d in combinations(connector_nodes,2):
		if graph[d[0],d[1]] > 0:
			connector_edges.append([d])
	connector_edges = np.array(connector_edges)
	degree_edges = np.array(degree_edges)
	idx = 0
	de = len(connector_edges)
	dmin = de - (de/10)
	dmax = de - (de/2)
	attack_degree_sps = []
	attack_pc_sps = []
	while True:
		num_kill = np.random.choice((dmin,dmax),1)
		np.random.shuffle(degree_edges)
		np.random.shuffle(connector_edges)
		d_edges = degree_edges[:num_kill]
		c_edges = connector_edges[:num_kill]
		d_graph = graph.copy()
		c_graph = graph.copy()
		delete_edges = []
		for node1,node2 in d_edges.reshape(d_edges.shape[0],2):
			delete_edges.append(graph.get_eid(node1,node2))
		d_graph.delete_edges(delete_edges)
		if d_graph.is_connected() == False:
				break
		delete_edges = []
		for node1,node2 in c_edges.reshape(c_edges.shape[0],2):
			delete_edges.append(graph.get_eid(node1,node2))
		c_graph.delete_edges(delete_edges)
		if c_graph.is_connected() == False:
				break
		deg_sp = np.array(d_graph.shortest_paths()).astype(float)
		c_sp = np.array(c_graph.shortest_paths()).astype(float)
		attack_degree_sps.append(np.nansum(deg_sp))
		attack_pc_sps.append(np.nansum(c_sp))
		idx = idx + 1
		if idx == 5000:
			break
	return [attack_pc_sps,attack_degree_sps,healthy_sp]

def human_attacks():
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
	df = pd.DataFrame(columns=['Attack Type','Sum of Shortest Paths'])
	btw_df = pd.DataFrame(columns=['Betweenness','Node Type','Task'])
	for task in tasks:
		atlas = 'power'
		print task
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		matrices = static_results['matrices']
		variables = []
		for i in np.arange(20):
			cost = .2
			temp_matrix = np.nanmean(static_results['matrices'],axis=0).copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=cost,mst=True)
			variables.append(graph)
		pool = Pool(20)
		results = pool.map(attack,variables)
		attack_degree_sps = np.array([])
		attack_pc_sps = np.array([])
		healthy_sp = np.array([])
		for r in results:
			attack_pc_sps = np.append(r[0],attack_pc_sps)
			attack_degree_sps = np.append(r[1],attack_degree_sps)
			healthy_sp = np.append(r[2],healthy_sp)
		attack_degree_sps = np.array(attack_degree_sps).reshape(-1)
		attack_pc_sps = np.array(attack_pc_sps).reshape(-1)
		print scipy.stats.ttest_ind(attack_pc_sps,attack_degree_sps)
		print scipy.stats.ttest_ind(attack_pc_sps,healthy_sp)
		pool = Pool(20)
		b_results = pool.map(between_community_centrality,variables)
		pc_btw = []
		deg_btw = []
		for r in b_results:
			pc_btw = np.append(r[0],pc_btw)
			deg_btw = np.append(r[1],deg_btw)	
		print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(pc_btw,deg_btw)	
		sys.stdout.flush()
		hdf = pd.DataFrame()
		task_str = np.ones(len(healthy_sp)).astype(str)
		task_str[:] = task
		hdf['Task'] = task_str
		hdf['Sum of Shortest Paths'] = healthy_sp
		hdf['Attack Type'] = 'None'
		df = df.append(hdf)
		d_df = pd.DataFrame()
		task_str = np.ones(len(attack_degree_sps)).astype(str)
		task_str[:] = task
		d_df['Task'] = task_str
		d_df['Sum of Shortest Paths'] = attack_degree_sps
		d_df['Attack Type'] = 'Degree Rich Club'
		df = df.append(d_df)
		pc_df = pd.DataFrame()
		task_str = np.ones(len(attack_pc_sps)).astype(str)
		task_str[:] = task
		pc_df['Task'] = task_str
		pc_df['Sum of Shortest Paths'] = attack_pc_sps
		pc_df['Attack Type'] = 'PC Rich Club'
		df = df.append(pc_df)
		tbtw_df = pd.DataFrame()
		tbtw_df['Betweenness'] = pc_btw
		tbtw_df['Node Type'] = 'PC'
		task_str = np.ones(len(pc_btw)).astype(str)
		task_str[:] = task
		tbtw_df['Task'] = task_str
		btw_df = btw_df.append(tbtw_df)
		tbtw_df = pd.DataFrame()
		task_str = np.ones(len(deg_btw)).astype(str)
		task_str[:] = task
		tbtw_df['Task'] = task_str
		tbtw_df['Betweenness'] = deg_btw
		tbtw_df['Node Type'] = 'Degree'
		btw_df = btw_df.append(tbtw_df)
		df.to_csv('/home/despoB/mb3152/dynamic_mod/results/human_attack')
		btw_df.to_csv('/home/despoB/mb3152/dynamic_mod/results/human_bwt')
	df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/human_attack')
	btw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/human_bwt')
	colors= sns.color_palette(['#3F6075', '#FFC61E'])
	sns.set(style="whitegrid",font_scale=2)
	sns.plt.figure(figsize=(51.2,22.8))
	sns.boxplot(data=df[df['Attack Type']!='None'],x='Task',hue='Attack Type',y="Sum of Shortest Paths",palette=colors)
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/attack_human.pdf',dpi=3600)
	sns.plt.close()
	sns.plt.figure(figsize=(51.2,22.8))
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	sns.boxplot(data=btw_df,x='Task',hue='Node Type',y="Betweenness",palette=colors)
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/humanbtw.pdf',dpi=3600)
	sns.plt.show()   
	scipy.stats.ttest_ind(df['Sum of Shortest Paths'][df['Attack Type'] == 'Degree Rich Club'],df['Sum of Shortest Paths'][df['Attack Type'] == 'None'])
	scipy.stats.ttest_ind(df['Sum of Shortest Paths'][df['Attack Type'] == 'PC Rich Club'],df['Sum of Shortest Paths'][df['Attack Type'] == 'Degree Rich Club'])

def rich_club_intersect(graph,rank):
	vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	pc = vc.pc
	mem = np.array(vc.community.membership)
	deg = vc.community.graph.strength(weights='weight')
	deg = np.argsort(deg)[rank:]
	pc = np.argsort(pc)[rank:]
	try:
		overlap = len(deg)/float(len(np.intersect1d(pc,deg)))
	except:
		overlap = 0
	return [overlap,len(np.unique(mem[pc]))/float(len(np.unique(mem))),len(np.unique(mem[deg]))/float(len(np.unique(mem)))]

def human_rich_club():
	"""
	rich club stuff
	"""
	df = pd.DataFrame(columns=["Percent Overlap", 'Percent Community, PC','Percent Community, Degree','Task'])
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
	for task in tasks:
		atlas = 'power'
		print task
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		matrices = static_results['matrices']
		avg_pc_normalized_phis = []
		avg_degree_normalized_phis = []
		intersects = []
		for cost in np.arange(5,21)*0.01:
			temp_matrix = np.nanmean(static_results['matrices'],axis=0).copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=cost,mst=True)
			temp_matrix = np.nanmean(static_results['matrices'],axis=0).copy()
			inters = rich_club_intersect(graph,211)
			temp_df = pd.DataFrame(columns=["Percent Overlap", 'Percent Community, PC','Percent Community, Degree','Task'],index=np.arange(1))
			temp_df["Percent Overlap"] = inters[0]
			temp_df['Percent Community, PC'] = inters[1]
			temp_df['Percent Community, Degree'] = inters[2]
			temp_df['Task'] = task
			df = df.append(temp_df)
			# if cost == .1:
			# 	vc = graph.community_infomap(edge_weights='weight',trials=500)
			# 	if task == 'WM':
			# 		membership = np.array(pd.read_csv('/home/despoB/mb3152/human_wm.csv').community.values)
			# 		np.fill_diagonal(temp_matrix,0.0)	
			# 		plot_corr_matrix(temp_matrix,membership,return_array=False,out_file='/home/despoB/mb3152/dynamic_mod/figures/%s_corr_mat_fin.pdf'%(task),label=False)		
			# 	else:
			# 		np.fill_diagonal(temp_matrix,0.0)
			# 		vc = graph.community_infomap(edge_weights='weight')
			# 		plot_corr_matrix(temp_matrix,vc.membership,return_array=False,out_file='/home/despoB/mb3152/dynamic_mod/figures/%s_corr_mat_fin.pdf'%(task),label=False)			
			
			degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
			average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=graph.strength(weights='weight')).phis() for i in range(100)],axis=0)
			degree_normalized_phis = degree_emperical_phis/average_randomized_phis
			vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
			pc = vc.pc
			pc[np.isnan(pc)] = 0.0
			pc_emperical_phis = RC(graph, scores=pc).phis()
			pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=pc).phis() for i in range(100)],axis=0)
			pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
			
			# graph.vs['pc'] = pc
			# graph.vs['Community'] = vc.community.membership
			# pc_rc = np.array(pc)>np.percentile(pc,80)
			# graph.vs['pc_rc'] = pc_rc
			# degree_rc = np.array(graph.strength(weights='weight')) > np.percentile(graph.strength(weights='weight'),80)
			# graph.vs['degree_rc'] = degree_rc
			# 1/0
			# graph.write_gml('human_gephi.gml')


			# import matlab
			# eng = matlab.engine.start_matlab()
			# eng.addpath('/home/despoB/mb3152/BrainNet/')
			# write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
			# write_df[3] = graph.strength(weights='weight')
			# write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/%s_degree.node'%(task),sep='\t',index=False,names=False,header=False)
			# node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/%s_degree.node'%(task)
			# surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
			# img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/%s_degree.png' %(task)
			# configs = '/home/despoB/mb3152/BrainNet/pc_values.mat'
			# eng.BrainNet_MapCfg(node_file,surf_file,img_file,configs)

			# write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
			# write_df[3] = pc
			# write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/%s_pc_avg.node'%(task),sep='\t',index=False,names=False,header=False)
			# node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/%s_pc_avg.node'%(task)
			# surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
			# img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/%s_pc_avg.png' %(task)
			# configs = '/home/despoB/mb3152/BrainNet/pc_values.mat'
			# eng.BrainNet_MapCfg(node_file,surf_file,img_file,configs)
			avg_pc_normalized_phis.append(pc_normalized_phis)
			avg_degree_normalized_phis.append(degree_normalized_phis)
		sns.set_style("white")
		sns.set_style("ticks")
		with sns.plotting_context("paper",font_scale=1):	
			sns.tsplot(np.array(avg_degree_normalized_phis)[:,:-2],color='b',condition='Degree',ci=95)
			sns.tsplot(np.array(avg_pc_normalized_phis)[:,:-2],color='r',condition='PC',ci=95)
			plt.ylabel('Normalized Rich Club Coefficient')
			plt.xlabel('Rank')
			sns.despine()
			plt.legend()
			plt.tight_layout()
			plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_%s.pdf'%(task),dpi=3600)
			plt.close()
	df["Percent Overlap"] = df["Percent Overlap"].astype(float)
	df['Percent Community, PC'] = df['Percent Community, PC'].astype(float)
	df['Percent Community, Degree'] = df['Percent Community, Degree'].astype(float)
	sns.barplot(data=df,x='Percent Overlap',y='Task')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_overlap_human.pdf')
	sns.plt.show()
	sns.barplot(data=df,x='Percent Community, PC',y='Task')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_pc.pdf')
	sns.plt.show()
	sns.barplot(data=df,x='Percent Community, Degree',y='Task')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_degree.pdf')	
	sns.plt.show()
	for task in tasks:
		print scipy.stats.ttest_ind(df['Percent Community, PC'][df.Task==task],df['Percent Community, Degree'][df.Task == task])

def classifier(t):
	train = np.ones(len(pc_vals)).astype(bool)
	train[t] = False
	print 'fitting pc/Q model'
	clf = linear_model.BayesianRidge()
	clf.fit(pc_vals[train],task_perf[train])
	pc_prediction = clf.predict(pc_vals[t])
	print 'fitting edge model'
	clf = linear_model.BayesianRidge()
	clf.fit(fit_matrices[train],task_perf[train])
	matrix_prediction = clf.predict(fit_matrices[t])
	return pc_prediction,matrix_prediction

def pc_classifier(t):
	train = np.ones(len(pc_vals)).astype(bool)
	train[t] = False
	clf = linear_model.BayesianRidge()
	clf.fit(pc_vals[train],task_perf[train])
	pc_prediction = clf.predict(pc_vals[t])
	return pc_prediction

def predict(v):
	pvals = v[0]
	train = v[1] 
	t = v[2]
	task_perf = v[3] 
	train = np.ones(len(train)).astype(bool)
	train[t] = False
	clf = linear_model.LinearRegression()
	clf.fit(pvals[train],task_perf[train])
	return clf.predict(pvals[t])

def corrfunc(x, y, **kws):
	r, _ = pearsonr(x, y)
	ax = plt.gca()
	ax.annotate("r={:.3f}".format(r) + ",p={:.3f}".format(_),xy=(.1, .9), xycoords=ax.transAxes)

def plot_connectivity_results(data,x,y,save_str):
	sns.set_style("white")
	sns.set_style("ticks")
	colors = sns.palettes.color_palette('Paired',7)
	colors = np.array(colors)
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(data, col='Task', hue='Task',sharex=False,sharey=False,palette=colors,col_wrap=4)
		g = g.map(sns.regplot,x,y,scatter_kws={'alpha':.50})
		g.map(corrfunc,x,y)
		sns.despine()
		plt.tight_layout()
		plt.savefig(save_str,dpi=3600)
		plt.close()

def plot_results(data,x,y,save_str):
	sns.set_style("white")
	sns.set_style("ticks")
	colors = np.array(sns.palettes.color_palette('Paired',6))
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(data, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,x,y,scatter_kws={'alpha':.95})
		g.map(corrfunc,x,y)
		sns.despine()
		plt.tight_layout()
		plt.savefig(save_str,dpi=3600)
		plt.close()

def performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL']):
	tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL']
	project='hcp'
	atlas='power'
	known_membership,network_names,num_nodes,name_int_dict = network_labels('power')
	df = pd.DataFrame(columns=['PC','WCD','Task','PCxPerformance','PCxModularity','WCDxPerformance','WCDxModularity'])
	diff_df = pd.DataFrame(columns=['Task','Modularity_Type','Performance'])
	task_perf_df_cols = ['Task','Modularity Increasing Diversity Value','Performance']
	task_perf_df = pd.DataFrame(columns=task_perf_df_cols)
	loo_columns= ['Task','Nodal Predicted Performance','Q Predicted Performance','Mean Nodal Predicted Performance','Mean PC Predicted Performance','Mean WCD Predicted Performance','Performance','Mod Diff','PC Q+ Diff','WCD Q+ Diff','PC Q+/Q- Diff','WCD Q+/Q- Diff','Rest Versus Task']
	loo_df = pd.DataFrame(columns = loo_columns)
	for task in tasks:
		"""
		see which graph metrics correlate with modularity and performance
		"""
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,'REST',atlas)
		assert (subjects == np.load('/home/despoB/mb3152/dynamic_mod/results/hcp_%s_%s_subs_fz.npy'%('REST',atlas))).all()
		rest_subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/hcp_%s_%s_subs_fz.npy'%('REST',atlas))
		rest_results = graph_metrics(rest_subjects,'REST',atlas,run_version='fz')
		rest_subject_pcs = rest_results['subject_pcs'].copy()
		rest_matrices = rest_results['matrices']
		rest_subject_mods = rest_results['subject_mods']
		rest_subject_wmds = rest_results['subject_wmds']
		print task
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		assert (subjects == np.load('/home/despoB/mb3152/dynamic_mod/results/hcp_%s_%s_subs_fz.npy'%(task,atlas))).all()
		subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/hcp_%s_%s_subs_fz.npy'%(task,atlas))
		static_results = graph_metrics(subjects,task,atlas,run_version='fz')
		subject_pcs = static_results['subject_pcs'].copy()
		matrices = static_results['matrices']
		subject_mods = static_results['subject_mods']
		subject_wmds = static_results['subject_wmds']
		rest_idx = []
		for i,s in enumerate(rest_subjects):
			if s not in subjects:
				continue
			rest_idx.append(i)
		task_perf = task_performance(subjects,task)
		if task == 'SOCIAL':
			task_perf[task_perf==1] = np.nan
			# task_perf[task_perf==0.91666666666675001] = np.nan
		assert subject_pcs.shape[0] == len(subjects)
		mean_pc = np.nanmean(static_results['subject_pcs'],axis=0)
		df_array = []
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
		mod_wmd_corr = np.zeros(subject_wmds.shape[1])
		for i in range(subject_wmds.shape[1]):
			mod_wmd_corr[i] = nan_pearsonr(subject_mods,subject_wmds[:,i])[0]
		predict_nodes = np.where(mod_pc_corr>0.0)[0]
		local_predict_nodes = np.where(mod_pc_corr<0.0)[0]
		wmd_predict_nodes = np.where(mod_wmd_corr<0.0)[0]
		wmd_local_predict_nodes = np.where(mod_wmd_corr>0.0)[0]
		
		"""
		predict performance using high and low PCS values. 
		"""
		if task != 'REST':
			to_delete = np.isnan(task_perf).copy()
			to_delete = np.where(to_delete==True)
			subject_pcs = np.delete(subject_pcs,to_delete,axis=0)
			rest_subject_pcs = np.delete(rest_subject_pcs[rest_idx],to_delete,axis=0)
			rest_subject_wmds = np.delete(rest_subject_wmds[rest_idx],to_delete,axis=0)
			rest_subject_mods = np.delete(rest_subject_mods[rest_idx],to_delete,axis=0)
			matrices = np.delete(matrices,to_delete,axis=0)
			subject_mods = np.delete(subject_mods,to_delete)
			subject_wmds = np.delete(subject_wmds,to_delete,axis=0)
			task_perf = np.delete(task_perf,to_delete)
		task_perf = scipy.stats.zscore(task_perf)
		subject_pcs[np.isnan(subject_pcs)] = 0.0
		rest_subject_pcs[np.isnan(rest_subject_pcs)] = 0.0
		subject_wmds[np.isnan(subject_wmds)] = 0.0
		"""
		prediction / cross validation
		"""

		mod_diff = subject_mods - rest_subject_mods
		pvals = np.array(subject_mods - rest_subject_mods)
		pvals = pvals.reshape(pvals.shape[0],1)
		vs = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			vs.append([pvals,train,t,task_perf])
		pool = Pool(20)
		q_diff_prediction = pool.map(predict,vs)
		print 'Q Difference Prediction of Performance, LOO: ', pearsonr(q_diff_prediction,task_perf)

		mean_pc = []
		mean_local_pc = []
		mean_wmd = []
		mean_local_wmd = []
		for s in range(len(task_perf)):
			mean_pc.append(np.nanmean(subject_pcs[s,predict_nodes]))
			mean_local_pc.append(np.nanmean(subject_pcs[s,local_predict_nodes]))
			mean_wmd.append(np.nanmean(subject_wmds[s,wmd_predict_nodes]))
			mean_local_wmd.append(np.nanmean(subject_wmds[s,wmd_local_predict_nodes]))
		mean_pc = np.array(mean_pc)
		mean_local_pc = np.array(mean_local_pc)
		mean_wmd = np.array(mean_wmd)
		mean_local_wmd = np.array(mean_local_wmd)

		pvals = np.array([mean_pc-mean_local_pc,mean_local_wmd-mean_wmd]).transpose()
		vs = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			vs.append([pvals,train,t,task_perf])
		pool = Pool(20)
		mean_nodal_prediction = pool.map(predict,vs)
		print 'Mean Nodal Prediction of Performance, LOO: ', pearsonr(np.array(mean_nodal_prediction).reshape(-1),task_perf)

		pvals = np.concatenate([subject_pcs,subject_wmds],axis=1)
		vs = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			vs.append([pvals,train,t,task_perf])
		pool = Pool(20)
		nodal_prediction = pool.map(predict,vs)
		print 'Nodal (PC and WMD scores) Prediction of Performance, LOO: ', pearsonr(nodal_prediction,task_perf)
		mod_diff = subject_mods - rest_subject_mods
		# pvals = np.array(subject_mods - rest_subject_mods)
		# pvals = pvals.reshape(pvals.shape[0],1)
		# vs = []
		# for t in range(pvals.shape[0]):
		# 	train = np.ones(len(pvals)).astype(bool)
		# 	train[t] = False
		# 	vs.append([pvals,train,t,task_perf])
		# pool = Pool(20)
		# q_diff_prediction = pool.map(predict,vs)
		print 'Q Difference Prediction of Performance: ', pearsonr(mod_diff,task_perf)

		# pvals = np.nanmean(subject_pcs[:,predict_nodes],axis=1)-np.nanmean(rest_subject_pcs[:,predict_nodes],axis=1)
		# pvals = pvals.reshape(pvals.shape[0],1)
		# vs = []
		# for t in range(pvals.shape[0]):
		# 	train = np.ones(len(pvals)).astype(bool)
		# 	train[t] = False
		# 	vs.append([pvals,train,t,task_perf])
		# pool = Pool(20)
		# q_plus_pc_diff_prediction = pool.map(predict,vs) 
		q_plus_pc_diff_prediction = np.nanmean(subject_pcs[:,predict_nodes],axis=1)-np.nanmean(rest_subject_pcs[:,predict_nodes],axis=1)
		print 'Q Plus PC Difference Prediction of Performance: ', pearsonr(q_plus_pc_diff_prediction,task_perf)

		# pvals = np.nanmean(subject_wmds[:,local_predict_nodes],axis=1)-np.nanmean(rest_subject_wmds[:,local_predict_nodes],axis=1)
		# pvals = pvals.reshape(pvals.shape[0],1)
		# vs = []
		# for t in range(pvals.shape[0]):
		# 	train = np.ones(len(pvals)).astype(bool)
		# 	train[t] = False
		# 	vs.append([pvals,train,t,task_perf])
		# pool = Pool(20)
		# q_plus_wmd_diff_prediction = pool.map(predict,vs) 
		q_plus_wmd_diff_prediction = np.nanmean(subject_wmds[:,local_predict_nodes],axis=1)-np.nanmean(rest_subject_wmds[:,local_predict_nodes],axis=1)
		print 'Q Plus WMD Difference Prediction of Performance: ', pearsonr(q_plus_wmd_diff_prediction,task_perf)

		# pvals = np.nanmean(subject_pcs[:,predict_nodes],axis=1)-np.nanmean(rest_subject_pcs[:,predict_nodes],axis=1)
		# pvals = pvals.reshape(pvals.shape[0],1)
		# vs = []
		# for t in range(pvals.shape[0]):
		# 	train = np.ones(len(pvals)).astype(bool)
		# 	train[t] = False
		# 	vs.append([pvals,train,t,mod_diff])
		# pool = Pool(20)
		# q_plus_pc_mod_diff_prediction = pool.map(predict,vs)
		q_plus_pc_mod_prediction = np.nanmean(subject_pcs[:,predict_nodes],axis=1)-np.nanmean(rest_subject_pcs[:,predict_nodes],axis=1)
		print 'Q Plus PC Difference Prediction of Mod Diff: ', pearsonr(q_plus_pc_mod_prediction,mod_diff)

		# pvals = np.nanmean(subject_wmds[:,local_predict_nodes],axis=1)-np.nanmean(rest_subject_wmds[:,local_predict_nodes],axis=1)
		# pvals = pvals.reshape(pvals.shape[0],1)
		# vs = []
		# for t in range(pvals.shape[0]):
		# 	train = np.ones(len(pvals)).astype(bool)
		# 	train[t] = False
		# 	vs.append([pvals,train,t,mod_diff])
		# pool = Pool(20)
		# q_plus_wmd_mod_diff_prediction = pool.map(predict,vs) 
		q_plus_wmd_mod_prediction = np.nanmean(subject_wmds[:,local_predict_nodes],axis=1)-np.nanmean(rest_subject_wmds[:,local_predict_nodes],axis=1)
		print 'Q Plus WMD Difference Prediction of Mod Diff: ', pearsonr(q_plus_wmd_mod_prediction,mod_diff)

		# pvals = subject_mods - rest_subject_mods
		# print 'Rest Q versus Task Q | Performance', pearsonr(pvals,task_perf)
		# print 'Q+ PC Rest V Task, Performance', pearsonr(task_perf,np.nanmean(subject_pcs[:,predict_nodes],axis=1)-np.nanmean(rest_subject_pcs[:,predict_nodes],axis=1))
		# print 'Q+ WCD Rest V Task, Performance', pearsonr(task_perf,np.nanmean(subject_wmds[:,wmd_local_predict_nodes],axis=1)-np.nanmean(rest_subject_wmds[:,wmd_local_predict_nodes],axis=1))
		# print 'Q+ PC Rest V Task, Modularity Rest V Task', pearsonr(pvals,np.nanmean(subject_pcs[:,predict_nodes],axis=1)-np.nanmean(rest_subject_pcs[:,predict_nodes],axis=1))
		# print 'Q+ WCD Rest V Task, Modularity Rest V Task',pearsonr(pvals,np.nanmean(subject_wmds[:,wmd_local_predict_nodes],axis=1)-np.nanmean(rest_subject_wmds[:,wmd_local_predict_nodes],axis=1))
		px = np.nanmean(subject_pcs[:,predict_nodes],axis=1)-np.nanmean(rest_subject_pcs[:,predict_nodes],axis=1)
		py = np.nanmean(subject_pcs[:,local_predict_nodes],axis=1)-np.nanmean(rest_subject_pcs[:,local_predict_nodes],axis=1)
		wx = np.nanmean(subject_wmds[:,wmd_local_predict_nodes],axis=1)-np.nanmean(rest_subject_wmds[:,wmd_local_predict_nodes],axis=1)
		wy = np.nanmean(subject_wmds[:,wmd_predict_nodes],axis=1)-np.nanmean(rest_subject_wmds[:,wmd_predict_nodes],axis=1)
		
		print 'PC, Q+/Q- Ratio Rest V Task, Performance', pearsonr(task_perf,px-py)
		print 'WCD, Q+/Q- Ratio Rest V Task, Performance', pearsonr(task_perf,wx-wy)
		print 'PC, Q+/Q- Ratio Rest V Task, Modularity Rest V Task', pearsonr(mod_diff,px-py)
		print 'WCD, Q+/Q- Ratio Rest V Task, Modularity Rest V Task',pearsonr(mod_diff,wx-wy)
		
		q_plus_pc_mod_diff_prediction  = px-py
		q_plus_wmd_mod_diff_prediction  = wx-wy
		pvals = np.zeros((len(wx),5))
		pvals[:,0] = px-py
		pvals[:,1] = wx-wy
		pvals[:,2] = subject_mods-rest_subject_mods
		pvals[:,3] = np.nanmean(subject_pcs[:,predict_nodes],axis=1)-np.nanmean(rest_subject_pcs[:,predict_nodes],axis=1)
		pvals[:,4] = np.nanmean(subject_wmds[:,wmd_local_predict_nodes],axis=1)-np.nanmean(rest_subject_wmds[:,wmd_local_predict_nodes],axis=1)
		vs = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			vs.append([pvals,train,t,task_perf])
		pool = Pool(20)
		five_diff_feat = pool.map(predict,vs)
		print 'Rest Task Difference (5 Feats) Prediction of Performance, LOO: ', pearsonr(five_diff_feat,task_perf)

		pvals = subject_mods
		pvals = pvals.reshape(pvals.shape[0],1)
		vs = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			vs.append([pvals,train,t,task_perf])
		pool = Pool(20)
		q_performance_prediction = pool.map(predict,vs)
		print 'Q Prediction of Performance, LOO: ', pearsonr(np.array(q_performance_prediction).reshape(-1),task_perf)
		
		for node in range(subject_pcs.shape[1]):
			df_array.append([np.nanmean(subject_pcs,axis=0)[node],np.nanmean(subject_wmds,axis=0)[node],task,nan_pearsonr(subject_pcs[:,node],task_perf)[0],mod_pc_corr[node],nan_pearsonr(subject_wmds[:,node],task_perf)[0],mod_wmd_corr[node]])
		df = pd.concat([df,pd.DataFrame(df_array,columns=['PC','WCD','Task','PCxPerformance','PCxModularity','WCDxPerformance','WCDxModularity'])],axis=0)		

		pvals = np.array([px-py,wx-wy]).transpose()
		vs = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			vs.append([pvals,train,t,task_perf])
		pool = Pool(20)
		mean_nodal_prediction_diff = pool.map(predict,vs)
		print 'Mean Nodal Difference Prediction of Performance, LOO: ', pearsonr(np.array(mean_nodal_prediction_diff).reshape(-1),task_perf)

		pvals = np.array([mean_pc-mean_local_pc]).transpose()
		vs = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			vs.append([pvals,train,t,task_perf])
		pool = Pool(20)
		mean_pc_nodal_prediction = pool.map(predict,vs)
		print 'Mean PC Prediction of Performance, LOO: ', pearsonr(np.array(mean_pc_nodal_prediction).reshape(-1),task_perf)

		pvals = np.array([mean_local_wmd-mean_wmd]).transpose()
		vs = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			vs.append([pvals,train,t,task_perf])
		pool = Pool(20)
		mean_wmd_nodal_prediction = pool.map(predict,vs)
		print 'Mean WMD Prediction of Performance, LOO: ', pearsonr(np.array(mean_wmd_nodal_prediction).reshape(-1),task_perf)
		loo_array = []
		for i in range(len(nodal_prediction)):
			loo_array.append([task,nodal_prediction[i],q_performance_prediction[i],mean_nodal_prediction[i],mean_pc_nodal_prediction[i],mean_wmd_nodal_prediction[i],task_perf[i],mod_diff[i],q_plus_pc_mod_prediction[i],q_plus_wmd_mod_prediction[i],q_plus_pc_mod_diff_prediction[i],q_plus_wmd_mod_diff_prediction[i],five_diff_feat[i]])
		loo_df = pd.concat([loo_df,pd.DataFrame(loo_array,columns=loo_columns)],axis=0)

	loo_df.to_csv('/home/despoB/mb3152/dynamic_mod/loo_df.csv')
	df.to_csv('/home/despoB/mb3152/dynamic_mod/df.csv')

	"""
	PRETTY FIGURES 
	"""
	loo_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/loo_df.csv')
	df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/df.csv')


	# plot_results(loo_df,'Nodal Predicted Performance','Performance','/home/despoB/mb3152/dynamic_mod/figures/Nodal_Predicted_Performance.pdf')
	# plot_results(loo_df,'Mean Nodal Predicted Performance','Performance','/home/despoB/mb3152/dynamic_mod/figures/Mean_Nodal_Predicted_Performance.pdf')
	# plot_results(loo_df,'Mean PC Predicted Performance','Performance','/home/despoB/mb3152/dynamic_mod/figures/PC_Predicted_Performance.pdf')
	# plot_results(loo_df,'Mean WCD Predicted Performance','Performance','/home/despoB/mb3152/dynamic_mod/figures/WCD_Predicted_Performance.pdf')
	# plot_results(loo_df,'Q Predicted Performance','Performance','/home/despoB/mb3152/dynamic_mod/figures/Q_Predicted_Performance.pdf')
	# plot_results(df,'PCxPerformance','PC','/home/despoB/mb3152/dynamic_mod/figures/PC_PC_Performance.pdf')
	# plot_results(df,'PCxPerformance','PCxModularity','/home/despoB/mb3152/dynamic_mod/figures/PC_Modularity_PC_Performance.pdf')
	# plot_results(df,'WCDxPerformance','WCD','/home/despoB/mb3152/dynamic_mod/figures/WCD_WCD_Performance.pdf')
	# plot_results(df,'WCDxPerformance','WCDxModularity','/home/despoB/mb3152/dynamic_mod/figures/WCD_Modularity_PC_Performance.pdf')
	# plot_results(loo_df,'Mod Diff','Performance','/home/despoB/mb3152/dynamic_mod/figures/Mod_Diff_Predicted_Performance.pdf')
	# plot_results(loo_df,'PC Q+ Diff','Performance','/home/despoB/mb3152/dynamic_mod/figures/PC_Q_Plus_Diff_Predicted_Performance.pdf')
	# plot_results(loo_df,'WCD Q+ Diff','Performance','/home/despoB/mb3152/dynamic_mod/figures/WCD_Q_Plus_Diff_Predicted_Performance.pdf')
	# plot_results(loo_df,'PC Q+ Diff','Mod Diff','/home/despoB/mb3152/dynamic_mod/figures/PC_Q_Plus_Diff_Predicted_Mod_Diff.pdf')
	# plot_results(loo_df,'WCD Q+ Diff','Mod Diff','/home/despoB/mb3152/dynamic_mod/figures/WCD_Q_Plus_Diff_Predicted_Mod_Diff.pdf')
	# plot_results(loo_df,'PC Q+/Q- Diff','Performance','/home/despoB/mb3152/dynamic_mod/figures/PC_Q_Plus_Minus_Diff_Predicted_Performance.pdf')
	# plot_results(loo_df,'WCD Q+/Q- Diff','Performance','/home/despoB/mb3152/dynamic_mod/figures/WCD_Q_Plus_Miuns_Diff_Predicted_Performance.pdf')
	# plot_results(loo_df,'PC Q+/Q- Diff','Mod Diff','/home/despoB/mb3152/dynamic_mod/figures/PC_Q_Plus_Minus_Diff_Predicted_Mod_Diff.pdf')
	# plot_results(loo_df,'WCD Q+/Q- Diff','Mod Diff','/home/despoB/mb3152/dynamic_mod/figures/WCD_Q_Plus_Minus_Diff_Predicted_Mod_Diff.pdf')
	# plot_results(loo_df,'Rest Versus Task','Performance','/home/despoB/mb3152/dynamic_mod/figures/Rest_Versus_Task_Predicted_Performance.pdf')
	
def all_attacks(graphs,out_file_name=None):
	attack_degree_sps = np.array([])
	attack_pc_sps = np.array([])
	healthy_sp = np.array([])
	pool = Pool(20)
	results = pool.map(attack,graphs)
	for r in results:
		attack_pc_sps = np.append(r[0],attack_pc_sps)
		attack_degree_sps = np.append(r[1],attack_degree_sps)
		healthy_sp = np.append(r[2],attack_degree_sps)
	print 'PC Versus Degree: ', scipy.stats.ttest_ind(attack_pc_sps,attack_degree_sps)
	print 'PC Versus Heathly, ', scipy.stats.ttest_ind(attack_pc_sps,healthy_sp)
	b_results = pool.map(between_community_centrality,graphs)
	pc_btw = []
	deg_btw = []
	for r in b_results:
		pc_btw = np.append(r[0],pc_btw)
		deg_btw = np.append(r[1],deg_btw)
	print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(pc_btw,deg_btw)
	df = pd.DataFrame()
	df['Sum of Shortest Paths'] = healthy_sp
	df['Attack Type'] = 'None'
	d_df = pd.DataFrame()
	d_df['Sum of Shortest Paths'] = attack_degree_sps
	d_df['Attack Type'] = 'Degree Rich Club'
	df = df.append(d_df)
	pc_df = pd.DataFrame()
	pc_df['Sum of Shortest Paths'] = attack_pc_sps
	pc_df['Attack Type'] = 'PC Rich Club'
	df = df.append(pc_df)
	sns.violinplot(data=df,y='Sum of Shortest Paths',x='Attack Type',inner='quartile')
	sns.plt.show()

def attack_an_animal(animal='Worms'):
	if animal == 'Worms':
		worms = ['Worm1','Worm2','Worm3','Worm4']
		variables = []
		for worm in worms:
			matrix = np.array(pd.read_excel('pnas.1507110112.sd01.xls',sheetname=worm).corr())[4:,4:]
			for cost in np.arange(5,21)*0.01:
				temp_matrix = matrix.copy()
				graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=cost,mst=True)
				variables.append(graph)
		all_attacks(variables)

def c_elegans_rich_club(plt_mat=False):
	worms = ['Worm1','Worm2','Worm3','Worm4']
	for worm in worms:
		# matrix = np.arctanh(np.array(pd.read_excel('pnas.1507110112.sd01.xls',sheetname=worm).corr())[4:,4:])
		matrix = np.array(pd.read_excel('pnas.1507110112.sd01.xls',sheetname=worm).corr())[4:,4:]
		avg_pc_normalized_phis = []
		avg_degree_normalized_phis = []
		for cost in np.arange(5,21)*0.01:
			temp_matrix = matrix.copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=cost,mst=True)
			if cost == .1:
				if plt_mat == True:
					np.fill_diagonal(matrix,0.0)
					vc = graph.community_infomap(edge_weights='weight',trials=500)
					plot_corr_matrix(matrix,vc.membership,return_array=False,out_file='/home/despoB/mb3152/dynamic_mod/figures/%s_corr_mat.pdf'%(worm),label=False)
			if graph.is_connected() == False:
				continue
			1/0
			degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
			average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=graph.strength(weights='weight')).phis() for i in range(500)],axis=0)
			degree_normalized_phis = degree_emperical_phis/average_randomized_phis
			avg_degree_normalized_phis.append(degree_normalized_phis)
			vc = graph.community_infomap(edge_weights='weight',trials=500)
			pc = brain_graphs.brain_graph(vc).pc
			pc[np.isnan(pc)] = 0.0
			# if worm == 'Worm4':
			# 	if cost == .2
			# 	graph.vs['pc'] = pc
			# 	graph.vs['Community'] = vc.membership
			# 	rc_pc = np.array(pc)>.5
			# 	graph.vs['rc_rc'] = rc_pc
			# 	degree_rc = np.array(graph.strength(weights='weight')) > 8.7
			# 	graph.vs['degree_rc'] = degree_rc
			# 	graph.write_gml('ce_gephi.gml')
			pc_emperical_phis = RC(graph, scores=pc).phis()
			pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=pc).phis() for i in range(500)],axis=0)
			pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
			avg_pc_normalized_phis.append(pc_normalized_phis)
		sns.set_style("white")
		sns.set_style("ticks")
		# degree_normalized_phis = np.nanmean(avg_degree_normalized_phis,axis=0)
		# pc_normalized_phis = np.nanmean(avg_pc_normalized_phis,axis=0)
		with sns.plotting_context("paper",font_scale=1):	
			sns.tsplot(np.array(avg_degree_normalized_phis)[:,:-2],color='b',condition='Degree',ci=95)
			sns.tsplot(np.array(avg_pc_normalized_phis)[:,:-2],color='r',condition='PC',ci=95)
			plt.ylabel('Normalized Rich Club Coefficient')
			plt.xlabel('Rank')
			sns.despine()
			plt.legend()
			plt.tight_layout()
			plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_%s.pdf'%(worm),dpi=3600)
			plt.close()

def plot_reordered_matrix(matrix,membership=None):
	import matlab
	eng = matlab.engine.start_matlab()
	eng.addpath('/home/despoB/mb3152/brain_graphs/BCT/2016_01_16_BCT/')
	r_matrix = eng.reorderMAT(matlab.double(matrix.tolist()),10,'line')
	if type(membership) != 'NoneType':
		membership = np.array(membership) + 1
		m_matrix = eng.reorder_mod(r_matrix,matlab.double(membership.tolist()))

def power_rich_club():
	graph = Graph.Read_GML('power.gml')
	graph.es["weight"] = np.ones(graph.ecount())
	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=graph.strength(weights='weight')).phis() for i in range(50)],axis=0)
	degree_normalized_phis = degree_emperical_phis/average_randomized_phis
	vc = graph.community_infomap(edge_weights='weight',trials=25)
	pc = brain_graphs.brain_graph(vc).pc
	pc[np.isnan(pc)] = 0.0
	# graph.vs['pc'] = pc
	# graph.vs['Community'] = vc.membership
	# pc_rc = np.array(pc)>np.percentile(pc,80)
	# graph.vs['pc_rc'] = pc_rc
	# degree_rc = np.array(graph.strength(weights='weight')) > np.percentile(graph.strength(weights='weight'),80)
	# graph.vs['degree_rc'] = degree_rc
	# graph.write_gml('power_gephi.gml')
	pc_emperical_phis = RC(graph, scores=pc).phis()
	pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=pc).phis() for i in range(50)],axis=0)
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	sns.set_style("white")
	sns.set_style("ticks")
	with sns.plotting_context("paper",font_scale=1):	
		sns.tsplot(np.array(degree_normalized_phis)[:-2],color='b',condition='Degree',ci=99)
		sns.tsplot(np.array(pc_normalized_phis)[:-2],color='r',condition='PC',ci=99)
		plt.ylabel('Normalized Rich Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend()
		plt.tight_layout()
		plt.show()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_power.pdf',dpi=3600)
		plt.show()

def c_elegans_str_rich_club():
	graph = Graph.Read_GML('celegansneural.gml')
	graph.es["weight"] = np.ones(graph.ecount())
	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=graph.strength(weights='weight')).phis() for i in range(100)],axis=0)
	degree_normalized_phis = degree_emperical_phis/average_randomized_phis
	vc = graph.community_infomap(edge_weights='weight',trials=1000)
	pc = brain_graphs.brain_graph(vc).pc
	pc[np.isnan(pc)] = 0.0
	# graph.vs['pc'] = pc
	# graph.vs['Community'] = vc.membership
	# pc_rc = np.array(pc)>np.percentile(pc,80)
	# graph.vs['pc_rc'] = pc_rc
	# degree_rc = np.array(graph.strength(weights='weight')) > np.percentile(graph.strength(weights='weight'),80)
	# graph.vs['degree_rc'] = degree_rc
	# graph.write_gml('struc_ce_gephi.gml')
	pc_emperical_phis = RC(graph, scores=pc).phis()
	pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=pc).phis() for i in range(1000)],axis=0)
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	sns.set_style("white")
	sns.set_style("ticks")
	with sns.plotting_context("paper",font_scale=1):	
		sns.tsplot(np.array(degree_normalized_phis)[:-2],color='b',condition='Degree',ci=99)
		sns.tsplot(np.array(pc_normalized_phis)[:-2],color='r',condition='PC',ci=99)
		plt.ylabel('Normalized Rich Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_structural.pdf',dpi=3600)
		plt.show()

def cat_and_macaque_rich_club(animal='cat'):
	if animal == 'macaque':
		matrix = loadmat('%s.mat'%(animal))['CIJ']
	else:
		matrix = loadmat('%s.mat'%(animal))['CIJall']
	graph = brain_graphs.matrix_to_igraph(matrix,cost=1.)
	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	degree = graph.strength(weights='weight')
	average_randomized_phis = np.mean([RC(preserve_strength(graph,randomize_topology=True),scores=degree).phis() for i in range(5000)],axis=0)
	degree_normalized_phis = degree_emperical_phis/average_randomized_phis
	graph = brain_graphs.matrix_to_igraph(matrix,cost=1.)
	vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	pc = vc.pc
	pc[np.isnan(pc)] = 0.0
	pc_emperical_phis = RC(graph, scores=pc).phis()
	pc_average_randomized_phis = np.mean([RC(preserve_strength(graph,randomize_topology=True),scores=pc).phis() for i in range(5000)],axis=0)
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	sns.set_style("white")
	sns.set_style("ticks")
	# if animal == 'macaque':
	# 	graph.vs['pc'] = pc
	# 	graph.vs['Community'] = vc.community.membership
	# 	pc_rc = np.array(pc)>np.percentile(pc,80)
	# 	graph.vs['pc_rc'] = pc_rc
	# 	degree_rc = np.array(graph.strength(weights='weight')) > np.percentile(graph.strength(weights='weight'),80)
	# 	graph.vs['degree_rc'] = degree_rc
	# 	graph.write_gml('macaque_gephi.gml')
	with sns.plotting_context("paper",font_scale=1):	
		sns.tsplot(degree_normalized_phis[:-2],color='b',condition='Degree',ci=90)
		sns.tsplot(pc_normalized_phis[:-2],color='r',condition='PC',ci=90)
		plt.ylabel('Normalized Rich Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_%s.pdf'%(animal),dpi=3600)
		plt.show()
		# plt.close()

def airlines_RC():
	vs = []
	sources = pd.read_csv('routes.dat',header=None)[3].values
	dests = pd.read_csv('routes.dat',header=None)[5].values
	graph = Graph()
	for s in sources:
		if s in dests:
			continue
		try:
			vs.append(int(s))
		except:
			continue
	for s in dests:
		try:
			vs.append(int(s))
		except:
			continue

	graph.add_vertices(np.unique(vs).astype(str))
	sources = pd.read_csv('routes.dat',header=None)[3].values
	dests = pd.read_csv('routes.dat',header=None)[5].values
	for s,d in zip(sources,dests):
		try:
			int(s)
			int(d)
		except:
			continue
		if int(s) not in vs:
			continue
		if int(d) not in vs:
			continue
		s = str(s)
		d = str(d)
		eid = graph.get_eid(s,d,error=False)
		if eid == -1:
			graph.add_edge(s,d,weight=1)
		else:
			graph.es[eid]['weight'] = graph.es[eid]["weight"] + 1
	graph.delete_vertices(np.argwhere((np.array(graph.degree())==0)==True))
	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	degree = graph.strength(weights='weight')
	average_randomized_phis = np.mean([RC(preserve_strength(graph,randomize_topology=True),scores=degree).phis() for i in range(10)],axis=0)
	degree_normalized_phis = degree_emperical_phis/average_randomized_phis
	vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	pc = vc.pc
	pc[np.isnan(pc)] = 0.0
	pc_emperical_phis = RC(graph, scores=pc).phis()
	pc_average_randomized_phis = np.mean([RC(preserve_strength(graph,randomize_topology=True),scores=pc).phis() for i in range(10)],axis=0)
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	sns.set_style("white")
	sns.set_style("ticks")
	airports = pd.read_csv('airports.dat',header=None)
	with sns.plotting_context("paper",font_scale=1):	
		sns.tsplot(np.array(degree_normalized_phis)[:-2],color='b',condition='Degree',ci=99)
		sns.tsplot(np.array(pc_normalized_phis)[:-2],color='r',condition='PC',ci=99)
		plt.ylabel('Normalized Rich Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend()
		plt.tight_layout()
		plt.show()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_airports.pdf',dpi=3600)
		plt.show()
	pc_int = []
	degree_int = []
	for i in range(2,1000):
		num_int = 0
		for i in range(1,i):
		    if 'Intl' in airports[1][airports[0]==int(graph.vs['name'][np.argsort(pc)[-i]])].values[0]:
		    	num_int = num_int + 1
		print num_int
		pc_int.append(num_int)
		num_int = 0
		for i in range(1,i):
		    if 'Intl' in airports[1][airports[0]==int(graph.vs['name'][np.argsort(graph.strength(weights='weight'))[-i]])].values[0]:
		    	num_int = num_int + 1
		print num_int
		degree_int.append(num_int)
	with sns.plotting_context("paper",font_scale=1):	
		sns.tsplot(degree_int,color='b',condition='Degree',ci=99)
		sns.tsplot(pc_int,color='r',condition='PC',ci=99)
		plt.ylabel('International Airports in Rich Club')
		plt.xlabel('Airports')
		sns.despine()
		plt.legend()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_airports_int.pdf',dpi=3600)
		plt.show()

	longitudes = []
	latitudes = []
	for v in range(graph.vcount()):
		latitudes.append(airports[6][airports[0]==int(graph.vs['name'][v])])
		longitudes.append(airports[7][airports[0]==int(graph.vs['name'][v])])

# targeted_attacks()
# performance_across_tasks(atlas='power',tasks=[sys.argv[1]])
# performance_across_tasks()
# split_connectivity_across_tasks()
# human_attacks()
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
		subjects = remove_missing_subjects(list(np.array(hcp_subjects).copy()),sys.argv[2],sys.argv[3])
		graph_metrics(subjects,task=sys.argv[2],atlas=sys.argv[3],run=True)
	if sys.argv[1] == 'make_matrix':
		subject = str(sys.argv[2])
		task = str(sys.argv[3])
		atlas = str(sys.argv[4])
		make_static_matrix(subject,task,'hcp',atlas)

"""
Methods

Resting State: ICA FIX plus whole brain signal and Bandpass filter, motion
Tasks: cerebral spinal fluid signal, white matter signal, whole brain signal, Bandpass Filter, motion

Results

Figure 1: Analysis Explanation. Single Node's PC across subjects

Figure 2: PC by Diversity Facilitated Modularity Coefficient, all tasks.

Figure 3a: PC by Diversity Facilitated Performance Coefficient, all tasks with performance values.
Figure 3b: Diversity Facilitated Modularity Coefficient by Diversity Facilitated Performance Coefficient.

Figure 4: Performance Prediction Figure, median split and correlations.

Figure 5a: PC edge corr analyses of types of changes
Figure 5b: Correlation between the stength of an edge (i,j) and how the PC of i facilitates the connectivity changes of j.

Figure 6: Rich Connector Club. PC results in a higher normalized phi, linear pattern.

"""

