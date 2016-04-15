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
	if label==False:
		y_names = []
		x_names = []
	else:
		ax.set_yticks(x_ticks)
		ax.set_xticks(y_ticks)
	sns.heatmap(corr_mat,square=True,yticklabels=y_names,xticklabels=x_names,vmin=-.3,vmax=.3,linewidths=0.0,cmap="RdBu_r")
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

def connectivity_across_tasks(atlas='power',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL']):
	global hcp_subjects
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL']
	project='hcp'
	atlas = 'power'
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	df_columns=['Participation Coefficient','Within-Community-Degree','Task','Diversity Facilitated Modularity Coefficient','Provinciality Facilitated Modularity Coefficient','PC-Q Coefficients','WMD-Q Coefficients']
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
		print 'Diversity Facilitated Modularity Coefficient, Mean PC: ', nan_pearsonr(mod_pc_corr,mean_pc)

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
			df_array.append([mean_pc[node],mean_wmd[node],task,mod_pc_corr[node],mod_wmd_corr[node],clf.coef_[264+node],clf.coef_[node]])
		df = pd.concat([df,pd.DataFrame(df_array,columns=df_columns)],axis=0)
		
		q_prediction = []
		for t in range(len(pvals)):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			clf = linear_model.BayesianRidge()
			clf.fit(pvals[train],subject_mods[train])
			q_prediction.append(clf.predict(pvals[t]))
		print 'Nodal Role Prediction of Q, LOO: ', pearsonr(q_prediction,subject_mods)
		loo_array = []
		for i in range(len(q_prediction)):
			loo_array.append([task,q_prediction[i],subject_mods[i]])
		loo_df = pd.concat([loo_df,pd.DataFrame(loo_array,columns=loo_columns)],axis=0)


	"""
	make brain figures
	"""
	# 	#Figure 1
	# 	1/0
	# 	import matlab
	# 	eng = matlab.engine.start_matlab()
	# 	eng.addpath('/home/despoB/mb3152/BrainNet/')
	# 	target_node = 258
	# 	while True:
	# 		figure_subjects = np.random.randint(0,450,10)
	# 		if (np.argsort(subject_pcs[[figure_subjects],target_node]) == np.argsort(subject_mods[[figure_subjects]])).all():
	# 			break
	# 	for subject in figure_subjects:
	# 		write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
	# 		write_df[3] = np.nanmean(subject_pcs,axis=0)
	# 		write_df[4] = 3
	# 		write_df[4][target_node] = 5
	# 		write_df[3] = known_membership
	# 		write_df[known_membership == 0] = 0.0
	# 		write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/subject_%s.node'%(subject),sep='\t',index=False,names=False,header=False)
	# 		write_matrix = matrices[subject].copy()
	# 		write_matrix = brain_graphs.threshold(write_matrix,cost=0.15)
	# 		for i,j in combinations(range(264),2):
	# 			if i != target_node:
	# 				if j != target_node:
	# 					write_matrix[i,j] = 0.0
	# 					write_matrix[j,i] = 0.0
	# 		matrix_df = pd.DataFrame(write_matrix)
	# 		matrix_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/subject_%s.edge'%(subject),sep='\t',index=False,names=False,header=False)
	# 		node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/subject_%s.node'%(subject)
	# 		edge_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/subject_%s.edge'%(subject)
	# 		surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
	# 		img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/figure_1_%s_mod(%s)_pc(%s).png' %(str(subject),str(np.round(subject_mods[subject],2)),str(np.round(subject_pcs[subject][258],2)))
	# 		configs = '/home/despoB/mb3152/BrainNet/pc_values_edges.mat'
	# 		eng.BrainNet_MapCfg(node_file,edge_file,surf_file,img_file,configs)
	# 	np.save('/home/despoB/mb3152/dynamic_mod/brain_figures/figure_1_data.npy',np.array([figure_subjects,subject_pcs[figure_subjects,target_node],subject_mods[figure_subjects]]))
	# 	# figure 2
	# 	write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
	# 	pcs = np.nanmean(subject_pcs,axis=0)
	# 	write_df[3] = pcs
	# 	# maxv = np.nanmean(pcs) + (np.nanstd(pcs)*2.75)
	# 	# minv = np.nanmean(pcs) - (np.nanstd(pcs)*2.75)
	# 	# write_df[3][pcs > maxv] = maxv
	# 	# write_df[3][pcs < minv] = minv	
	# 	write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_%s.node'%(task),sep='\t',index=False,names=False,header=False)
	# 	node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_%s.node'%(task)
	# 	surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
	# 	img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/figure_2_pc_%s.png' %(task)
	# 	configs = '/home/despoB/mb3152/BrainNet/pc_values.mat'
	# 	eng.BrainNet_MapCfg(node_file,surf_file,configs,img_file)
	# 	#mod pc values
	# 	write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
	# 	write_df[3] = mod_pc_corr
	# 	maxv = np.nanmean(mod_pc_corr) + (np.nanstd(mod_pc_corr)*2.75)
	# 	minv = np.nanmean(mod_pc_corr) - (np.nanstd(mod_pc_corr)*2.75)
	# 	write_df[3][mod_pc_corr > maxv] = maxv
	# 	write_df[3][mod_pc_corr < minv] = minv
	# 	write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_mod_%s.node'%(task),sep='\t',index=False,names=False,header=False)			
	# 	node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_mod_%s.node'%(task)
	# 	surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
	# 	img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/figure_2_mod_pc_corr_%s.png' %(task)
	# 	configs = '/home/despoB/mb3152/BrainNet/pc_values.mat'
	# 	eng.BrainNet_MapCfg(node_file,surf_file,configs,img_file)

	sns.set_style("white")
	sns.set_style("ticks")
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(loo_df, col='Task', hue='Task',sharex=True,sharey=True,palette='Paired',col_wrap=3)
		g = g.map(sns.regplot,'Predicted Q','Q',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/WMD_PC_Q_Prediction.pdf',dpi=3600)
		plt.show()

	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=True,sharey=True,palette='Paired',col_wrap=3)
		g = g.map(sns.regplot,'Participation Coefficient','Diversity Facilitated Modularity Coefficient',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/PCxPCxModularity.pdf',dpi=3600)
		plt.show()

	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=True,sharey=True,palette='Paired',col_wrap=3)
		g = g.map(sns.regplot,'Within-Community-Degree','Provinciality Facilitated Modularity Coefficient',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/WMDxWMDxModularity.pdf',dpi=3600)
		plt.show()

	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=True,sharey=True,palette='Paired',col_wrap=3)
		g = g.map(sns.regplot,'Participation Coefficient','PC-Q Coefficients',scatter_kws={'alpha':.95})
		plt.ylim(np.min(df['PC-Q Coefficients']),np.max(df['PC-Q Coefficients']))
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/PC-Q_Coefs.pdf',dpi=3600)
		plt.show()

	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=True,sharey=True,palette='Paired',col_wrap=3)
		g = g.map(sns.regplot,'Within-Community-Degree','WMD-Q Coefficients',scatter_kws={'alpha':.95})
		plt.ylim(np.min(df['WMD-Q Coefficients']),np.max(df['WMD-Q Coefficients']))
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/WMD-Q_Coefs.pdf',dpi=3600)
		plt.show()

	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=True,sharey=True,palette='Paired',col_wrap=3)
		g = g.map(sns.regplot,'Within-Community-Degree','Participation Coefficient',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/WMDxPC.pdf',dpi=3600)
		plt.show()


	#Brain images of PC and Diversity Facilitated Modularity Change
	"""
	Make a matrix of each node's PC correlation to all edges in the graph.
	"""
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL']
	project='hcp'
	atlas = 'power'
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
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
		"""
		for pc
		"""
		# predict_nodes = np.where(mod_pc_corr>0.0)[0]
		# local_predict_nodes = np.where(mod_pc_corr<0.0)[0]
		# pc_edge_corr = np.arctanh(pc_edge_correlation(subject_pcs,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr_z.npy' %(project,task,atlas)))
		"""
		for wcd
		"""
		predict_nodes = np.where(mod_wmd_corr>0.0)[0]
		local_predict_nodes = np.where(mod_wmd_corr<0.0)[0]
		pc_edge_corr = np.arctanh(pc_edge_correlation(subject_wmds,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_wmd_edge_corr_z.npy' %(project,task,atlas)))

		# only consider edges that are present in the graph if you take the top half of correlations. 
		edge_thresh = 50
		edge_thresh = np.percentile(np.nanmean(matrices,axis=0),edge_thresh)
		pc_edge_corr[:,np.nanmean(matrices,axis=0)<edge_thresh] = np.nan
		pc_thresh = 75
		local_thresh = 25
		pc_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),pc_thresh)
		local_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),local_thresh)
		connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)>=pc_thresh)[0]
		local_nodes = np.where(np.nanmean(subject_pcs,axis=0)<local_thresh)[0]
		
		pc_edge_corr[:,np.nanmean(matrices,axis=0)<edge_thresh,] = 0.0
		high_pc_edge_matrix = np.nanmean(pc_edge_corr[predict_nodes],axis=0)
		low_pc_edge_matrix = np.nanmean(pc_edge_corr[local_predict_nodes],axis=0)

		## Plot matrix of changes
		# diff_pc_edge_matrix = np.diff([np.nanmean(pc_edge_corr[connector_nodes],axis=0),np.nanmean(pc_edge_corr[local_nodes],axis=0)],axis=0).reshape((264,264))
		# matrix = (np.tril(low_pc_edge_matrix) + np.triu(high_pc_edge_matrix)).reshape((264,264))
		# plot_corr_matrix(matrix,network_names.copy(),out_file=None,plot_corr=True,return_array=False)

		#Within and between network edge PC modulation weights in matrix, absolute,positive, and negative
		pc_edge_corr_pos = pc_edge_corr.copy()
		pc_edge_corr_neg = pc_edge_corr.copy()
		pc_edge_corr_abs = np.abs(pc_edge_corr.copy())
		pc_edge_corr_pos[pc_edge_corr_pos<0.0]=0.0
		pc_edge_corr_neg[pc_edge_corr_neg>0.0]=0.0
		pc_edge_corr_neg = np.abs(pc_edge_corr_neg)
		
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
		violin_columns = ["Changes","Node Type","Edge Type"]
		task_violin_df = pd.DataFrame(columns=violin_columns)

		# result_array_to_add = pc_edge_corr[connector_within_network_mask].reshape(-1)[pc_edge_corr[connector_within_network_mask].reshape(-1)!=0.0]
		# edge_type_ = make_strs_for_df(result_array_to_add,'Within Sub-Network, All')
		# node_type_ = make_strs_for_df(result_array_to_add,'Connector')
		# df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		# task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

		# result_array_to_add = pc_edge_corr[local_within_network_mask].reshape(-1)[pc_edge_corr[local_within_network_mask].reshape(-1)!=0.0]
		# edge_type_ = make_strs_for_df(result_array_to_add,'Within Sub-Network, All')
		# node_type_ = make_strs_for_df(result_array_to_add,'Local')
		# df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		# task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)
		
		# task_violin_df.Changes = task_violin_df.Changes.astype(float)

		# print 'Within Sub-Network, All: ' + str(scipy.stats.ttest_ind(task_violin_df.Changes[task_violin_df['Node Type']=='Connector'][task_violin_df['Edge Type']=='Within Sub-Network, All'],
		# 	task_violin_df.Changes[task_violin_df['Node Type']=='Local'][task_violin_df['Edge Type']=='Within Sub-Network, All']))
		
		# result_array_to_add = pc_edge_corr[connector_between_network_mask].reshape(-1)[pc_edge_corr[connector_between_network_mask].reshape(-1)!=0.0]
		# edge_type_ = make_strs_for_df(result_array_to_add,'Between Sub-Network, All')
		# node_type_ = make_strs_for_df(result_array_to_add,'Connector')
		# df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		# task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

		# result_array_to_add = pc_edge_corr[local_between_network_mask].reshape(-1)[pc_edge_corr[local_between_network_mask].reshape(-1)!=0.0]
		# edge_type_ = make_strs_for_df(result_array_to_add,'Between Sub-Network, All')
		# node_type_ = make_strs_for_df(result_array_to_add,'Local')
		# df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		# task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)
		
		# task_violin_df.Changes = task_violin_df.Changes.astype(float)

		# print 'Between Sub-Network, All: ' + str(scipy.stats.ttest_ind(task_violin_df.Changes[task_violin_df['Node Type']=='Connector'][task_violin_df['Edge Type']=='Between Sub-Network, All'],
		# 	task_violin_df.Changes[task_violin_df['Node Type']=='Local'][task_violin_df['Edge Type']=='Between Sub-Network, All']))

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

		result_array_to_add = pc_edge_corr_pos[connector_between_network_mask].reshape(-1)[pc_edge_corr_pos[connector_between_network_mask].reshape(-1)>0]
		edge_type_ = make_strs_for_df(result_array_to_add,'Between Sub-Network, Positive')
		node_type_ = make_strs_for_df(result_array_to_add,'Connector')
		df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

		result_array_to_add = pc_edge_corr_pos[local_between_network_mask].reshape(-1)[pc_edge_corr_pos[local_between_network_mask].reshape(-1)>0]
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

		result_array_to_add = pc_edge_corr_neg[connector_between_network_mask].reshape(-1)[pc_edge_corr_neg[connector_between_network_mask].reshape(-1)>0]
		edge_type_ = make_strs_for_df(result_array_to_add,'Between Sub-Network, Negative')
		node_type_ = make_strs_for_df(result_array_to_add,'Connector')
		df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

		result_array_to_add = pc_edge_corr_neg[local_between_network_mask].reshape(-1)[pc_edge_corr_neg[local_between_network_mask].reshape(-1)>0]
		edge_type_ = make_strs_for_df(result_array_to_add,'Between Sub-Network, Negative')
		node_type_ = make_strs_for_df(result_array_to_add,'Local')
		df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)
		
		task_violin_df.Changes = task_violin_df.Changes.astype(float)
		
		print 'Between Sub-Network, Negative: ' + str(scipy.stats.ttest_ind(task_violin_df.Changes[task_violin_df['Node Type']=='Connector'][task_violin_df['Edge Type']=='Between Sub-Network, Negative'],
			task_violin_df.Changes[task_violin_df['Node Type']=='Local'][task_violin_df['Edge Type']=='Between Sub-Network, Negative']))

		# result_array_to_add = pc_edge_corr_abs[connector_within_network_mask].reshape(-1)[pc_edge_corr_abs[connector_within_network_mask].reshape(-1)>0]
		# edge_type_ = make_strs_for_df(result_array_to_add,'Within Sub-Network, Absolute')
		# node_type_ = make_strs_for_df(result_array_to_add,'Connector')
		# df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		# task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

		# result_array_to_add = pc_edge_corr_abs[local_within_network_mask].reshape(-1)[pc_edge_corr_abs[local_within_network_mask].reshape(-1)>0]
		# edge_type_ = make_strs_for_df(result_array_to_add,'Within Sub-Network, Absolute')
		# node_type_ = make_strs_for_df(result_array_to_add,'Local')
		# df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		# task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)
		
		# task_violin_df.Changes = task_violin_df.Changes.astype(float)
		
		# print 'Within Sub-Network, Absolute: ' + str(scipy.stats.ttest_ind(task_violin_df.Changes[task_violin_df['Node Type']=='Connector'][task_violin_df['Edge Type']=='Within Sub-Network, Absolute'],
		# 	task_violin_df.Changes[task_violin_df['Node Type']=='Local'][task_violin_df['Edge Type']=='Within Sub-Network, Absolute']))

		# result_array_to_add = pc_edge_corr_abs[connector_between_network_mask].reshape(-1)[pc_edge_corr_abs[connector_between_network_mask].reshape(-1)>0]
		# edge_type_ = make_strs_for_df(result_array_to_add,'Between Sub-Network, Absolute')
		# node_type_ = make_strs_for_df(result_array_to_add,'Connector')
		# df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		# task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)

		# result_array_to_add = pc_edge_corr_abs[local_between_network_mask].reshape(-1)[pc_edge_corr_abs[local_between_network_mask].reshape(-1)>0]
		# edge_type_ = make_strs_for_df(result_array_to_add,'Between Sub-Network, Absolute')
		# node_type_ = make_strs_for_df(result_array_to_add,'Local')
		# df_array_to_add = make_array_for_df([result_array_to_add,node_type_,edge_type_])
		# task_violin_df = task_violin_df.append(pd.DataFrame(data=df_array_to_add,columns=violin_columns),ignore_index=True)
		
		# task_violin_df.Changes = task_violin_df.Changes.astype(float)
		
		# print 'Between Sub-Network, Absolute: ' + str(scipy.stats.ttest_ind(task_violin_df.Changes[task_violin_df['Node Type']=='Connector'][task_violin_df['Edge Type']=='Between Sub-Network, Absolute'],
		# 	task_violin_df.Changes[task_violin_df['Node Type']=='Local'][task_violin_df['Edge Type']=='Between Sub-Network, Absolute']))

		# append for average of all task
		violin_df = violin_df.append(pd.DataFrame(data=task_violin_df,columns=violin_columns),ignore_index=True)
		# Figure for single Task
		sns.set_style("white")
		sns.set_style("ticks")
		colors = sns.color_palette(['#fdfd96','#C4D8E2'])
		with sns.plotting_context("paper",font_scale=2):
			sns.violinplot(palette = {'Connector': colors[0],'Local':colors[1]},x="Edge Type", y="Changes", hue="Node Type",order=['Within Sub-Network, Positive','Within Sub-Network, Negative'], data=task_violin_df,inner="quart",split=True,cut=0)
			sns.despine()
			plt.tight_layout()
			plt.ylim(0.,.3)
			plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/pc_edge_mod_%s.pdf'%(task),dpi=4600)

	print 'Within Sub-Network, Positive: ' + str(scipy.stats.ttest_ind(violin_df.Changes[violin_df['Node Type']=='Connector'][violin_df['Edge Type']=='Within Sub-Network, Positive'],
		violin_df.Changes[violin_df['Node Type']=='Local'][violin_df['Edge Type']=='Within Sub-Network, Positive']))
	print 'Within Sub-Network, Negative: ' + str(scipy.stats.ttest_ind(violin_df.Changes[violin_df['Node Type']=='Connector'][violin_df['Edge Type']=='Within Sub-Network, Negative'],
		violin_df.Changes[violin_df['Node Type']=='Local'][violin_df['Edge Type']=='Within Sub-Network, Negative']))
	print 'Between Sub-Network, Positive: ' + str(scipy.stats.ttest_ind(violin_df.Changes[violin_df['Node Type']=='Connector'][violin_df['Edge Type']=='Between Sub-Network, Positive'],
		violin_df.Changes[violin_df['Node Type']=='Local'][violin_df['Edge Type']=='Between Sub-Network, Positive']))
	print 'Between Sub-Network, Negative: ' + str(scipy.stats.ttest_ind(violin_df.Changes[violin_df['Node Type']=='Connector'][violin_df['Edge Type']=='Between Sub-Network, Negative'],
		violin_df.Changes[violin_df['Node Type']=='Local'][violin_df['Edge Type']=='Between Sub-Network, Negative']))
	# Average of All
	sns.set_style("white")
	sns.set_style("ticks")
	colors = sns.color_palette(['#fdfd96','#C4D8E2'])
	with sns.plotting_context("paper",font_scale=2):
		sns.violinplot(palette = {'Connector': colors[0],'Local':colors[1]},x="Edge Type", y="Changes", hue="Node Type",order=['Within Sub-Network, Positive','Within Sub-Network, Negative','Between Sub-Network, Positive','Between Sub-Network, Negative'], data=violin_df,inner="quart",split=True,cut=0)
		plt.ylim(0.,.3)
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/wmd_edge_mod_avg.pdf',dpi=4600)
		plt.close()
	
	"""
	Specificity of modulation by nodes' PC.
	Does the PC value of i impact the connectivity of j as i and j are more strongly connected?
	"""
	atlas = 'power'
	project='hcp'
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	for task in tasks:
		pc_thresh = 75
		local_thresh = 25
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		subject_wmds = static_results['subject_wmds']
		matrices = static_results['matrices']
		# pc_edge_corr = pc_edge_correlation(subject_pcs,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr_z.npy' %(project,task,atlas))
		# pc_thresh = np.percentile(np.nanmean(subject_pcs,axis=0),pc_thresh)
		pc_edge_corr = pc_edge_correlation(subject_wmds,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_wmd_edge_corr_z.npy' %(project,task,atlas))
		pc_thresh = np.percentile(np.nanmean(subject_wmds,axis=0),pc_thresh)
		# connector_nodes = np.where(np.nanmean(subject_pcs,axis=0)>=pc_thresh)[0]
		# local_nodes = np.where(np.nanmean(subject_pcs,axis=0)<=local_thresh)[0]
		connector_nodes = np.where(np.nanmean(subject_wmds,axis=0)>=pc_thresh)[0]
		local_nodes = np.where(np.nanmean(subject_wmds,axis=0)<=local_thresh)[0]
		#sum of weight changes for each node, by each node.
		driver_nodes_list = ['connector_nodes','local_nodes']
		edge_thresh = 50.0
		edge_thresh = np.percentile(np.nanmean(matrices,axis=0),edge_thresh)
		pc_edge_corr[:,np.nanmean(matrices,axis=0)<edge_thresh] = np.nan
		for driver_nodes in driver_nodes_list:
			weight_change_matrix = np.zeros((num_nodes,num_nodes))
			weight_change_matrix_pos = np.zeros((num_nodes,num_nodes))
			weight_change_matrix_neg = np.zeros((num_nodes,num_nodes))
			if driver_nodes == 'local_nodes':
				driver_nodes_array = local_nodes
			else:
				driver_nodes_array = connector_nodes
			for n1,n2 in permutations(range(num_nodes),2):
				if n1 not in driver_nodes_array:
					continue
				mask = np.ones((num_nodes),dtype=bool)
				mask[n1] = False
				array = pc_edge_corr[n1][n2]
				masked_array = array[mask]
				weight_change_matrix[n1,n2] = np.nansum(np.abs(masked_array))
				weight_change_matrix_pos[n1,n2] = abs(np.nansum(masked_array[masked_array>0]))
				weight_change_matrix_neg[n1,n2] = abs(np.nansum(masked_array[masked_array<0]))
			print driver_nodes
			temp_matrix = np.nanmean(matrices,axis=0)
			sns.set_style("white")
			sns.set_style("ticks")
			weight_matrix = weight_change_matrix
			r=pearsonr(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1))
			r = np.round(r[0],3),np.round(r[1],3)
			print pearsonr(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1))
			assert np.max(abs(np.diagonal(weight_matrix))) == 0.0
			with sns.plotting_context("paper",font_scale=1):
				sns.regplot(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1),color='Black',scatter=True,scatter_kws={'alpha':.5},label = r)
				plt.xlabel("Abosolute Sum of j's Diversity Faciliated Connectivity Changes by i")
				plt.ylabel('Edge Weight Between Nodes i and j')
				plt.legend(loc='best')
				plt.tight_layout()
				plt.show()
				plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/%s_all_connectivity_%s.jpeg'%(task,str(driver_nodes)),dpi=3600)
				plt.close()

	"""
	Are connector nodes modulating the edges that are most variable across subjects?
	"""
	atlas='power'
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	for task in tasks:
		pc_thresh = 75
		local_thresh = 25
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		matrices = static_results['matrices']
		pc_edge_corr = pc_edge_correlation(subject_pcs,thresh_matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr.npy' %(project,task,atlas))
		std_mod = []
		for i in range(num_nodes):
			std_mod.append(nan_pearsonr(pc_edge_corr[i].reshape(-1),np.std(matrices,axis=0).reshape(-1))[0])
		print task, pearsonr(np.nanmean(subject_pcs,axis=0),std_mod)
		plot_corr_matrix(np.std(matrices,axis=0),network_names.copy(),out_file=None,plot_corr=True,return_array=False)

	"""
	rich club stuff
	"""
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL']
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

		for cost in np.arange(5,21)*0.01:
			temp_matrix = np.nanmean(static_results['matrices'],axis=0).copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=cost,mst=True)
			temp_matrix = np.nanmean(static_results['matrices'],axis=0).copy()
			if cost == .1:
				vc = graph.community_infomap(edge_weights='weight',trials=500)
				if task == 'WM':
					membership = np.array(pd.read_csv('/home/despoB/mb3152/human_wm.csv').community.values)
					np.fill_diagonal(temp_matrix,0.0)	
					plot_corr_matrix(temp_matrix,membership,return_array=False,out_file='/home/despoB/mb3152/dynamic_mod/figures/%s_corr_mat_fin.pdf'%(task),label=False)		
				else:
					np.fill_diagonal(temp_matrix,0.0)
					vc = graph.community_infomap(edge_weights='weight')
					plot_corr_matrix(temp_matrix,vc.membership,return_array=False,out_file='/home/despoB/mb3152/dynamic_mod/figures/%s_corr_mat_fin.pdf'%(task),label=False)			
			
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


			import matlab
			eng = matlab.engine.start_matlab()
			eng.addpath('/home/despoB/mb3152/BrainNet/')
			write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
			write_df[3] = graph.strength(weights='weight')
			write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/%s_degree.node'%(task),sep='\t',index=False,names=False,header=False)
			node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/%s_degree.node'%(task)
			surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
			img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/%s_degree.png' %(task)
			configs = '/home/despoB/mb3152/BrainNet/pc_values.mat'
			eng.BrainNet_MapCfg(node_file,surf_file,img_file,configs)

			write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
			write_df[3] = pc
			write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/%s_pc_avg.node'%(task),sep='\t',index=False,names=False,header=False)
			node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/%s_pc_avg.node'%(task)
			surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
			img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/%s_pc_avg.png' %(task)
			configs = '/home/despoB/mb3152/BrainNet/pc_values.mat'
			eng.BrainNet_MapCfg(node_file,surf_file,img_file,configs)
			avg_pc_normalized_phis.append(pc_normalized_phis)
			avg_degree_normalized_phis.append(degree_normalized_phis)
		sns.set_style("white")
		sns.set_style("ticks")
		with sns.plotting_context("paper",font_scale=1):	
			sns.tsplot(np.array(avg_degree_normalized_phis)[:,:-13],color='b',condition='Degree',ci=95)
			sns.tsplot(np.array(avg_pc_normalized_phis)[:,:-13],color='r',condition='PC',ci=95)
			plt.ylabel('Normalized Rich Club Coefficient')
			plt.xlabel('Rank')
			sns.despine()
			plt.legend()
			plt.tight_layout
			plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_%s.pdf'%(task),dpi=3600)
			plt.close()

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

def performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL']):
	tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL']
	project='hcp'
	atlas='power'
	known_membership,network_names,num_nodes,name_int_dict = network_labels('power')
	df = pd.DataFrame(columns=['PC','WCD','Task','PCxPerformance','PCxModularity','WCDxPerformance','WCDxModularity','PC_Coefs','WMD_Coefs'])
	diff_df = pd.DataFrame(columns=['Task','Modularity_Type','Performance'])
	task_perf_df_cols = ['Task','Modularity Increasing Diversity Value','Performance']
	task_perf_df = pd.DataFrame(columns=task_perf_df_cols)
	loo_columns= ['Task','Nodal Predicted Performance','Q Predicted Performance','Mean Nodal Predicted Performance','Mean PC Predicted Performance','Mean WCD Predicted Performance','Performance']
	loo_df = pd.DataFrame(columns = loo_columns)
	for task in tasks:
		"""
		see which graph metrics correlate with modularity and performance
		"""
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
		task_perf = task_performance(subjects,task)
		assert subject_pcs.shape[0] == len(subjects)
		mean_pc = np.nanmean(static_results['subject_pcs'],axis=0)
		df_array = []
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
		mod_wmd_corr = np.zeros(subject_wmds.shape[1])
		for i in range(subject_wmds.shape[1]):
			mod_wmd_corr[i] = nan_pearsonr(subject_mods,subject_wmds[:,i])[0]
		
		"""
		predict performance using high and low PCS values. 
		"""
		if task != 'REST':
			to_delete = np.isnan(task_perf).copy()
			to_delete = np.where(to_delete==True)
			subject_pcs = np.delete(subject_pcs,to_delete,axis=0)
			subject_mods = np.delete(subject_mods,to_delete)
			subject_wmds = np.delete(subject_wmds,to_delete,axis=0)
			task_perf = np.delete(task_perf,to_delete)
		task_perf = scipy.stats.zscore(task_perf)
		subject_pcs[np.isnan(subject_pcs)] = 0.0
		subject_wmds[np.isnan(subject_wmds)] = 0.0


		"""
		prediction / cross validation
		"""
		# pvals = subject_wmds
		# pvals = subject_pcs
		pvals = np.concatenate([subject_pcs,subject_wmds],axis=1)
		clf = linear_model.BayesianRidge()
		clf.fit(pvals,task_perf)
		print 'PC Prediction of Performance: ', pearsonr(clf.predict(pvals),task_perf)
		print 'WMD, Coefficients of Performance', pearsonr(np.nanmean(subject_wmds,axis=0),clf.coef_[264:])
		print 'PC, Coefficients of Performance', pearsonr(np.nanmean(subject_pcs,axis=0),clf.coef_[:264])
		
		for node in range(subject_pcs.shape[1]):
			df_array.append([np.nanmean(subject_pcs,axis=0)[node],np.nanmean(subject_wmds,axis=0)[node],task,nan_pearsonr(subject_pcs[:,node],task_perf)[0],mod_pc_corr[node],nan_pearsonr(subject_wmds[:,node],task_perf)[0],mod_wmd_corr[node],clf.coef_[node],clf.coef_[node*2]])
		df = pd.concat([df,pd.DataFrame(df_array,columns=['PC','WCD','Task','PCxPerformance','PCxModularity','WCDxPerformance','WCDxModularity','PC_Coefs','WMD_Coefs'])],axis=0)
		
		pvals = np.concatenate([subject_pcs,subject_wmds],axis=1)
		# nodal_prediction = np.ones(len(task_perf))
		nodal_prediction = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			clf = linear_model.BayesianRidge()
			clf.fit(pvals[train],task_perf[train])
			nodal_prediction.append(clf.predict(pvals[t]))
		print 'Nodal Prediction of Performance, LOO: ', pearsonr(nodal_prediction,task_perf)
		
		# q_performance_prediction = np.ones(len(task_perf))
		q_performance_prediction = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			clf = linear_model.BayesianRidge()
			clf.fit(subject_mods.reshape(len(subject_mods),1)[train],task_perf[train])
			q_performance_prediction.append(clf.predict(subject_mods[t]))
		print 'Q Prediction of Performance, LOO: ', pearsonr(np.array(q_performance_prediction).reshape(-1),task_perf)

		predict_nodes = np.where(mod_pc_corr>0.0)[0]
		local_predict_nodes = np.where(mod_pc_corr<0.0)[0]
		wmd_predict_nodes = np.where(mod_wmd_corr<0.0)[0]
		wmd_local_predict_nodes = np.where(mod_wmd_corr>0.0)[0]
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
		mean_nodal_prediction = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			clf = linear_model.BayesianRidge()
			clf.fit(pvals[train],task_perf[train])
			mean_nodal_prediction.append(clf.predict(pvals[t]))
		print 'Mean Nodal Prediction of Performance, LOO: ', pearsonr(np.array(mean_nodal_prediction).reshape(-1),task_perf)

		pvals = np.array([mean_pc-mean_local_pc]).transpose()
		mean_pc_nodal_prediction = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			clf = linear_model.BayesianRidge()
			clf.fit(pvals[train],task_perf[train])
			mean_pc_nodal_prediction.append(clf.predict(pvals[t]))
		print 'Mean PC Prediction of Performance, LOO: ', pearsonr(np.array(mean_pc_nodal_prediction).reshape(-1),task_perf)

		pvals = np.array([mean_local_wmd,mean_wmd]).transpose()
		mean_wmd_nodal_prediction = []
		for t in range(pvals.shape[0]):
			train = np.ones(len(pvals)).astype(bool)
			train[t] = False
			clf = linear_model.BayesianRidge()
			clf.fit(pvals[train],task_perf[train])
			mean_wmd_nodal_prediction.append(clf.predict(pvals[t]))
		print 'Mean WMD Prediction of Performance, LOO: ', pearsonr(np.array(mean_wmd_nodal_prediction).reshape(-1),task_perf)

		loo_array = []
		for i in range(len(nodal_prediction)):
			loo_array.append([task,nodal_prediction[i],q_performance_prediction[i],mean_nodal_prediction[i],mean_pc_nodal_prediction[i],mean_wmd_nodal_prediction[i],task_perf[i]])
		loo_df = pd.concat([loo_df,pd.DataFrame(loo_array,columns=loo_columns)],axis=0)

		# diff = np.array(mean_pc)-np.array(mean_local_pc)
		# t_test = scipy.stats.ttest_ind(task_perf[np.argsort(diff)[len(diff)/2:]],task_perf[np.argsort(diff)[:len(diff)/2]])
		# diff_corr = nan_pearsonr(task_perf,np.array(mean_pc)-np.array(mean_local_pc))
		# high_mod_corr = nan_pearsonr(task_perf,np.array(mean_pc))
		# low_mod_corr = nan_pearsonr(task_perf,np.array(mean_local_pc))

		# print 't test, median split, t:', str(np.round(t_test[0],3)), 'p:', str(np.round(t_test[1],3))
		# print 'Correlation between [difference of connector and local PC scores], Performance, r:', str(np.round(diff_corr[0],3)), 'p:', str(np.round(diff_corr[1],3))
		# print 'Q, Performance, r:', pearsonr(subject_mods,task_perf)

		# array_len = len(diff)
		# str_list = np.chararray(array_len,itemsize=35)
		# str_list[:]= task
		# append_array = np.zeros((array_len,3)).astype(str)
		# append_array[:,0] = str_list
		# append_array[:,2] = scipy.stats.zscore(task_perf)[np.argsort(diff)]
		# append_array[:,1] = np.arange((array_len))[np.argsort(diff)].astype(str)
		# append_array[:,1][:array_len/2] = 'Locally Diverse'
		# append_array[:,1][array_len/2:] = 'Globally Diverse'
		# diff_df = diff_df.append(pd.DataFrame(data=append_array,columns=['Task','Modularity_Type','Performance']),ignore_index=True)
		
		# str_list = np.chararray(len(task_perf),itemsize=10)
		# str_list[:]= task
		# task_perf_array = np.zeros((len(task_perf),3)).astype(str)
		# task_perf_array[:,0] = str_list
		# task_perf_array[:,1] = np.array(mean_pc) - np.array(mean_local_pc)
		# task_perf_array[:,2] = scipy.stats.zscore(task_perf)
		# task_perf_df = task_perf_df.append([task_perf_df,pd.DataFrame(task_perf_array,columns=task_perf_df_cols)],ignore_index=True)
		
	
	# colors = sns.color_palette(['#fdfd96','#C4D8E2'])
	# sns.set(style="whitegrid", palette="pastel", color_codes=True)
	# with sns.plotting_context("paper"):
	# 	diff_df['Performance'] = diff_df['Performance'].astype(float)
	# 	sns.violinplot(x="Task", y="Performance", hue="Modularity_Type",data=diff_df,inner="quart",split=True,palette={'Globally Diverse': colors[0] , 'Locally Diverse': colors[1]})
	# 	sns.set()
	# 	sns.despine()
	# 	plt.tight_layout()
	# 	plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/task_performance.pdf',dpi=5600)
	# 	plt.show()

	# sns.set(style="whitegrid", palette="pastel", color_codes=True)
	# task_perf_df.Performance = task_perf_df.Performance.astype(float)
	# task_perf_df['Modularity Increasing Diversity Value'] = task_perf_df['Modularity Increasing Diversity Value'].astype(float)
	# colors = np.array(sns.palettes.color_palette('Paired',6))
	# with sns.plotting_context("paper",font_scale=1):
	# 	g = sns.FacetGrid(task_perf_df, hue='Task',col='Task', sharex=True,sharey=True,palette=colors[[0,2,4,5]])
	# 	g = g.map(sns.regplot,'Modularity Increasing Diversity Value','Performance',scatter_kws={'alpha':.5})
	# 	sns.despine()
	# 	plt.tight_layout()
	# 	plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/task_performance_corr.pdf',dpi=3600)
	# 	plt.show()
	sns.set_style("white")
	sns.set_style("ticks")
	colors = np.array(sns.palettes.color_palette('Paired',6))
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,'WMD_Coefs','WCD',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/WCD_Perf_Coefs.pdf',dpi=3600)
		plt.show()

	colors = np.array(sns.palettes.color_palette('Paired',6))
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,'PC_Coefs','PC',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/PC_Perf_Coefs.pdf',dpi=3600)
		plt.show()

	colors = np.array(sns.palettes.color_palette('Paired',6))
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(loo_df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,'Nodal Predicted Performance','Performance',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/Nodal_Predicted_Performance.pdf',dpi=3600)
		plt.show()

	colors = np.array(sns.palettes.color_palette('Paired',6))
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(loo_df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,'Mean Nodal Predicted Performance','Performance',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/Mean_Nodal_Predicted_Performance.pdf',dpi=3600)
		plt.show()

	colors = np.array(sns.palettes.color_palette('Paired',6))
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(loo_df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,'Mean PC Predicted Performance','Performance',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/PC_Predicted_Performance.pdf',dpi=3600)
		plt.show()

	colors = np.array(sns.palettes.color_palette('Paired',6))
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(loo_df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,'Mean WCD Predicted Performance','Performance',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/WCD_Predicted_Performance.pdf',dpi=3600)
		plt.show()

	colors = np.array(sns.palettes.color_palette('Paired',6))
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(loo_df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,'Q Predicted Performance','Performance',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/Q_Predicted_Performance.pdf',dpi=3600)
		plt.show()

	sns.set_style("white")
	sns.set_style("ticks")
	colors = np.array(sns.palettes.color_palette('Paired',6))
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,'PCxPerformance','PC',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/PC_PC_Performance.pdf',dpi=3600)
		plt.show()
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,'PCxPerformance','PCxModularity',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/PC_Modularity_PC_Performance.pdf',dpi=3600)
		plt.show()

	sns.set_style("white")
	sns.set_style("ticks")
	colors = np.array(sns.palettes.color_palette('Paired',6))
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,'WCDxPerformance','WCD',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/WCD_WCD_Performance.pdf',dpi=3600)
		plt.show()
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(df, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,'WCDxPerformance','WCDxModularity',scatter_kws={'alpha':.95})
		sns.despine()
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/WDC_Modularity_WCD_Performance.pdf',dpi=3600)
		plt.show()

def c_elegans_rich_club(plt_mat=False):
	worms = ['Worm1','Worm2','Worm3','Worm4']
	for worm in worms:
		matrix = np.arctanh(np.array(pd.read_excel('pnas.1507110112.sd01.xls',sheetname=worm).corr())[4:,4:])
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
			sns.tsplot(np.array(avg_degree_normalized_phis)[:,:-10],color='b',condition='Degree',ci=95)
			sns.tsplot(np.array(avg_pc_normalized_phis)[:,:-10],color='r',condition='PC',ci=95)
			plt.ylabel('Normalized Rich Club Coefficient')
			plt.xlabel('Rank')
			sns.despine()
			plt.legend()
			plt.tight_layout
			plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_%s.pdf'%(worm),dpi=3600)
			plt.show()
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
	vc = graph.community_infomap(edge_weights='weight',trials=1000)
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
		sns.tsplot(np.array(degree_normalized_phis)[:-50],color='b',condition='Degree',ci=99)
		sns.tsplot(np.array(pc_normalized_phis)[:-50],color='r',condition='PC',ci=99)
		plt.ylabel('Normalized Rich Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend()
		plt.tight_layout()
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
	pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=pc).phis() for i in range(100)],axis=0)
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	sns.set_style("white")
	sns.set_style("ticks")
	with sns.plotting_context("paper",font_scale=1):	
		sns.tsplot(np.array(degree_normalized_phis)[:-10],color='b',condition='Degree',ci=99)
		sns.tsplot(np.array(pc_normalized_phis)[:-10],color='r',condition='PC',ci=99)
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
	if animal == 'macaque':
		graph.vs['pc'] = pc
		graph.vs['Community'] = vc.community.membership
		pc_rc = np.array(pc)>np.percentile(pc,80)
		graph.vs['pc_rc'] = pc_rc
		degree_rc = np.array(graph.strength(weights='weight')) > np.percentile(graph.strength(weights='weight'),80)
		graph.vs['degree_rc'] = degree_rc
		graph.write_gml('macaque_gephi.gml')
	with sns.plotting_context("paper",font_scale=1):	
		sns.tsplot(degree_normalized_phis[:-10],color='b',condition='Degree',ci=90)
		sns.tsplot(pc_normalized_phis[:-10],color='r',condition='PC',ci=90)
		plt.ylabel('Normalized Rich Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend(loc='upper left')
		plt.tight_layout
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
	# with sns.plotting_context("paper",font_scale=1):	
	# 	sns.tsplot(np.array(degree_normalized_phis)[:-10],color='b',condition='Degree',ci=99)
	# 	sns.tsplot(np.array(pc_normalized_phis)[:-10],color='r',condition='PC',ci=99)
	# 	plt.ylabel('Normalized Rich Club Coefficient')
	# 	plt.xlabel('Rank')
	# 	sns.despine()
	# 	plt.legend()
	# 	plt.tight_layout()
	# 	plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_airports.pdf',dpi=3600)
	# 	plt.show()
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


performance_across_tasks(atlas='power',tasks=[sys.argv[1]])

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

