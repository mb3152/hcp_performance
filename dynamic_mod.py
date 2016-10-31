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
import matplotlib as mpl
from matplotlib import patches
plt.rcParams['pdf.fonttype'] = 42
path = '/home/despoB/mb3152/anaconda/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/Helvetica.ttf'
prop = mpl.font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = prop.get_name()
import seaborn as sns
import powerlaw
from richclub import preserve_strength, RC
from multiprocessing import Pool
sys.path.append('/home/despoB/mb3152/dynamic_mod/')
from complexity import FunctionalComplexity_Linear
from sklearn import linear_model, metrics
import random
global hcp_subjects
hcp_subjects = os.listdir('/home/despoB/connectome-data/')
hcp_subjects.sort()
global pc_vals 
global fit_matrices
global task_perf
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation
from scipy import stats, linalg
global homedir
# homedir = '/Users/Maxwell/HWNI/'
homedir = '/home/despoB/mb3152/'

def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr

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
	bdf = pd.read_csv('%s/dynamic_mod/os_behavior_data_900.csv'%(homedir))	
	for subject in subjects:
		try:
			files = glob.glob('%s/scanner_performance_data/%s_tfMRI_*%s*_Stats.csv' %(homedir,subject,task))
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

def plot_corr_matrix(matrix,membership,out_file=None,block_lower=False,return_array=True,plot_corr=True,label=True,reorder=True,colors=['None'],line=False,rectangle=False,draw_legend=False,colorbar=False):
	if reorder == True:
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
	else:
		corr_mat = matrix
		x_ticks = []
		y_ticks = []
		y_names = []
		x_names = []
		old_name = 0
		for i,name in enumerate(membership):
			if name == old_name:
				y_names.append(' ')
				x_names.append(' ')
				y_ticks.append(i)
				x_ticks.append(len(membership)-i)
				continue
			old_name = name
			y_ticks.append(i)
			x_ticks.append(len(membership)-i)
			y_names.append(name)
			x_names.append(name)
	sns.set(style='dark',context="paper",font='Helvetica',font_scale=1.2)
	# Set up the matplotlib figure
	# f, ax = plt.subplots(figsize=(12, 9))
	y_names.reverse()
	std = np.nanstd(corr_mat)
	mean = np.nanmean(corr_mat)
	vmin = mean - (std*2)
	vmax = mean + (std*2)
	if label != True:
		y_names = ['']
		x_names = ['']
	# cmap = sns.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
	# cmap=sns.diverging_palette(255,0,s=99,sep=80, n=15,center='dark',as_cmap=True)
	fig = sns.clustermap(corr_mat,yticklabels=y_names,xticklabels=x_names,cmap=sns.diverging_palette(260,10,sep=10, n=20,as_cmap=True),rasterized=True,col_colors=colors,row_colors=colors,row_cluster=False,col_cluster=False,**{'vmin':vmin,'vmax':vmax,'figsize':(15.567,15)})
	ax = fig.fig.axes[4]
	# Draw the heatmap using seaborn
	y_names.reverse()
	if block_lower:
		corr_mat = scipy.triu(corr_mat)
	ax.set_yticks(x_ticks)
	ax.set_xticks(y_ticks)
	y_names.reverse()
	if reorder == True:
		membership.sort()
	# Use matplotlib directly to emphasize known networks
	if line == True or rectangle == True:
		if len(colors) != len(membership):
			colors = np.arange(len(membership))
		for i,network,color, in zip(np.arange(len(membership)),membership,colors):
			if network != membership[i - 1]:
				if len(colors) != len(membership):
					color = 'white'
				if line == True:
					ax.axhline(len(membership) - i, c=color,linewidth=.5,label=network)
					ax.axhline(len(membership) - i, c='black',linewidth=.5)
					ax.axvline(i, c='black',linewidth=.5)
				if rectangle == True:
					ax.add_patch(patches.Rectangle((i+len(membership[membership==network]),264-i),len(membership[membership==network]),len(membership[membership==network]),facecolor="none",edgecolor=color,linewidth="1",angle=180))
	fig.ax_col_colors.add_patch(patches.Rectangle((0,0),264,1,facecolor="None",edgecolor='black',lw=2))
	fig.ax_row_colors.add_patch(patches.Rectangle((0,0),1,264,facecolor="None",edgecolor='black',lw=2))	
	col = fig.ax_col_colors.get_position()
	fig.ax_col_colors.set_position([col.x0, col.y0, col.width*1, col.height*.35])
	col = fig.ax_row_colors.get_position()
	fig.ax_row_colors.set_position([col.x0+col.width*(1-.35), col.y0, col.width*.35, col.height*1])
	fig.ax_col_dendrogram.set_visible(False)
	fig.ax_row_dendrogram.set_visible(False)
	if draw_legend == True:
		leg = fig.ax_heatmap.legend(bbox_to_anchor=[.98,1.1],ncol=5)
	for legobj in leg.legendHandles:
		legobj.set_linewidth(2.5)
	if colorbar == False:
		fig.cax.set_visible(False)
	if out_file != None:
		plt.savefig(out_file)
		plt.close()
	if plot_corr == True:
		plt.show()
	if return_array == True:
		plt.close()
		return corr_mat
	return fig

def make_static_matrix(subject,task,project,atlas,scrub=False):
	hcp_subject_dir = '/home/despoB/connectome-data/SUBJECT/*TASK*/*reg*'
	parcel_path = '/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas)
	MP = None
	# try:
	# 	MP = np.load('/home/despoB/mb3152/dynamic_mod/motion_files/%s_%s.npy' %(subject,task))
	# except:
	# 	run_fd(subject,task)
	# 	MP = np.load('/home/despoB/mb3152/dynamic_mod/motion_files/%s_%s.npy' %(subject,task))
	subject_path = hcp_subject_dir.replace('SUBJECT',subject).replace('TASK',task)
	if scrub == True:
		subject_time_series = brain_graphs.load_subject_time_series(subject_path,dis_file=MP,scrub_mm=0.2)
		brain_graphs.time_series_to_matrix(subject_time_series,parcel_path,voxel=False,fisher=False,out_file='/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_%s_matrix_scrubbed_0.2.npy' %(atlas,subject,atlas,task))
	if scrub == False:
		subject_time_series = brain_graphs.load_subject_time_series(subject_path,dis_file=None,scrub_mm=False)
		brain_graphs.time_series_to_matrix(subject_time_series,parcel_path,voxel=False,fisher=False,out_file='/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_%s_matrix.npy' %(atlas,subject,atlas,task))

def null_graph_individual_graph_analyes(matrix):
	cost = 0.05
	temp_matrix = matrix.copy()
	graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True)
	vc = graph.community_infomap(edge_weights='weight')
	temp_matrix = matrix.copy()
	random_matrix = temp_matrix.copy()
	random_matrix = random_matrix[np.tril_indices(264,-1)]
	np.random.shuffle(random_matrix)
	temp_matrix[np.tril_indices(264,-1)] = random_matrix
	temp_matrix[np.triu_indices(264)] = 0.0
	temp_matrix = np.nansum([temp_matrix,temp_matrix.transpose()],axis=0)
	graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True)
	graph = brain_graphs.brain_graph(VertexClustering(graph,vc.membership))
	return (graph.community.modularity,np.array(graph.pc),np.array(graph.wmd))

def null_community_individual_graph_analyes(matrix):
	cost = 0.05
	temp_matrix = matrix.copy()
	graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True)
	graph = graph.community_infomap(edge_weights='weight')
	membership = graph.membership
	np.random.shuffle(membership)
	graph = brain_graphs.brain_graph(VertexClustering(graph.graph,membership))
	return (graph.community.modularity,np.array(graph.pc),np.array(graph.wmd))

def null_all_individual_graph_analyes(matrix):
	cost = 0.01
	temp_matrix = matrix.copy()
	random_matrix = temp_matrix.copy()
	random_matrix = random_matrix[np.tril_indices(264,-1)]
	np.random.shuffle(random_matrix)
	temp_matrix[np.tril_indices(264,-1)] = random_matrix
	temp_matrix[np.triu_indices(264)] = 0.0
	temp_matrix = np.nansum([temp_matrix,temp_matrix.transpose()],axis=0)
	graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True)
	graph = graph.community_infomap(edge_weights='weight')
	graph = brain_graphs.brain_graph(graph)
	return (graph.community.modularity,np.array(graph.pc),np.array(graph.wmd))	

def individual_graph_analyes_wc(variables):
	subject = variables[0]
	print subject
	atlas = variables[1]
	task = variables[2]
	s_matrix = variables[3]	
	pc = []
	mod = []
	wmd = []
	memlen = []
	for cost in np.array(range(5,16))*0.01:
		temp_matrix = s_matrix.copy()
		graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True,mst=True)
		assert np.diff([cost,graph.density()])[0] < .01
		del temp_matrix
		graph = graph.community_infomap(edge_weights='weight')
		graph = brain_graphs.brain_graph(graph)
		pc.append(np.array(graph.pc))
		wmd.append(np.array(graph.wmd))
		mod.append(graph.community.modularity)
		memlen.append(len(graph.community.sizes()))
		del graph
	return (mod,np.nanmean(pc,axis=0),np.nanmean(wmd,axis=0),np.nanmean(memlen),subject)

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
		s_matrix = np.nanmean(s_matrix,axis=0)
		assert s_matrix.shape == (264,264)
		for cost in np.array(range(5,16))*0.01:
			temp_matrix = s_matrix.copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True)
			assert np.diff([cost,graph.density()])[0] < .005

def check_scrubbed_normalize(subjects,task,atlas='power'):	
	for subject in subjects:
		print subject
		s_matrix = []
		files = glob.glob('/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_*%s*_matrix_scrubbed_0.2.npy'%(atlas,subject,atlas,task))
		for f in files:
			dis_file = run_fd(subject,'_'.join(f.split('/')[-1].split('_')[2:5]))
			remove_array = np.zeros(len(dis_file))
			for i,fdf in enumerate(dis_file):
				if fdf > .2:
					remove_array[i] = 1
					if i == 0:
						remove_array[i+1] = 1
						continue
					if i == len(dis_file)-1:
						remove_array[i-1] = 1
						continue
					remove_array[i-1] = 1
					remove_array[i+1] = 1
			if len(remove_array[remove_array==1])/float(len(remove_array)) > .75:
				continue
			f = np.load(f)
			np.fill_diagonal(f,0.0)
			f[np.isnan(f)] = 0.0
			f = np.arctanh(f)
			s_matrix.append(f.copy())
		s_matrix = np.nanmean(s_matrix,axis=0)
		assert s_matrix.shape == (264,264)
		for cost in np.array(range(5,16))*0.01:
			temp_matrix = s_matrix.copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True)
			assert np.diff([cost,graph.density()])[0] < .005

def graph_metrics(subjects,task,atlas,run_version,project='hcp',run=False,scrubbed=False,homedir=homedir):
	"""
	run graph metrics or load them
	"""
	if run == False:
		# done_subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_%s.npy' %(project,task,atlas,run_version)) 
		# assert (done_subjects == subjects).all() #make sure you are getting subjects / subjects order you wanted and ran last time.
		subject_pcs = np.load('%sdynamic_mod/results/%s_%s_%s_pcs_%s.npy' %(homedir,project,task,atlas,run_version)) 
		subject_wmds = np.load('%sdynamic_mod/results/%s_%s_%s_wmds_%s.npy' %(homedir,project,task,atlas,run_version)) 
		subject_mods = np.load('%sdynamic_mod/results/%s_%s_%s_mods_%s.npy' %(homedir,project,task,atlas,run_version)) 
		try:
			subject_communities = np.load('%sdynamic_mod/results/%s_%s_%s_coms_%s.npy' %(homedir,project,task,atlas,run_version)) 
		except:
			print 'Loading alternative community numbers'
			subject_communities = np.load('%sdynamic_mod/results/%s_%s_%s_coms_fz_wc.npy' %(homedir,project,task,atlas)) 
		matrices = np.load('%sdynamic_mod/results/%s_%s_%s_matrices_%s.npy' %(homedir,project,task,atlas,run_version)) 
		thresh_matrices = np.load('%sdynamic_mod/results/%s_%s_%s_z_matrices_%s.npy' %(homedir,project,task,atlas,run_version))
		finished_subjects = np.load('%sdynamic_mod/results/%s_%s_%s_subs_%s.npy' %(homedir,project,task,atlas,run_version))
	elif run == True:
		finished_subjects = []
		variables = []
		matrices = []
		thresh_matrices = []
		for subject in subjects:
			s_matrix = []
			if scrubbed == True:
				files = glob.glob('%sdynamic_mod/%s_matrices/%s_%s_*%s*_matrix_scrubbed_0.2.npy'%(homedir,atlas,subject,atlas,task)) # FOR SCRUBBING ONLY
			if scrubbed == False:
				files = glob.glob('%sdynamic_mod/%s_matrices/%s_%s_*%s*_matrix.npy'%(homedir,atlas,subject,atlas,task))
			for f in files:
				if scrubbed == True:
					# FOR SCRUBBING ONLY
					dis_file = run_fd(subject,'_'.join(f.split('/')[-1].split('_')[2:5]))
					remove_array = np.zeros(len(dis_file))
					for i,fdf in enumerate(dis_file):
						if fdf > .2:
							remove_array[i] = 1
							if i == 0:
								remove_array[i+1] = 1
								continue
							if i == len(dis_file)-1:
								remove_array[i-1] = 1
								continue
							remove_array[i-1] = 1
							remove_array[i+1] = 1
					if len(remove_array[remove_array==1])/float(len(remove_array)) > .75:
						continue
				f = np.load(f)
				np.fill_diagonal(f,0.0)
				f[np.isnan(f)] = 0.0
				f = np.arctanh(f)
				s_matrix.append(f.copy())
			if len(s_matrix) == 0:
				continue
			s_matrix = np.nanmean(s_matrix,axis=0)
			variables.append([subject,atlas,task,s_matrix.copy()])
			num_nodes = s_matrix.shape[0]
			thresh_matrix = s_matrix.copy()
			thresh_matrix = scipy.stats.zscore(thresh_matrix.reshape(-1)).reshape((num_nodes,num_nodes))
			thresh_matrices.append(thresh_matrix.copy())
			matrices.append(s_matrix.copy())
			finished_subjects.append(subject)
		subject_mods = [] #individual subject modularity values
		subject_pcs = [] #subjects PCs
		subject_wmds = []
		subject_communities = []
		assert len(variables) == len(finished_subjects)
		print 'Running Graph Theory Analyses'
		from multiprocessing import Pool
		pool = Pool(18)
		results = pool.map(individual_graph_analyes_wc,variables)		
		for r,s in zip(results,finished_subjects):
			subject_mods.append(np.nanmean(r[0]))
			subject_pcs.append(r[1])
			subject_wmds.append(r[2])
			subject_communities.append(r[3])
			assert r[4] == s #make sure it returned the order of subjects/results correctly
		np.save('%sdynamic_mod/results/%s_%s_%s_pcs_%s.npy' %(homedir,project,task,atlas,run_version),np.array(subject_pcs))
		np.save('%sdynamic_mod/results/%s_%s_%s_wmds_%s.npy' %(homedir,project,task,atlas,run_version),np.array(subject_wmds))
		np.save('%sdynamic_mod/results/%s_%s_%s_mods_%s.npy' %(homedir,project,task,atlas,run_version),np.array(subject_mods))
		np.save('%sdynamic_mod/results/%s_%s_%s_subs_%s.npy' %(homedir,project,task,atlas,run_version),np.array(finished_subjects))
		np.save('%sdynamic_mod/results/%s_%s_%s_matrices_%s.npy'%(homedir,project,task,atlas,run_version),np.array(matrices))
		np.save('%sdynamic_mod/results/%s_%s_%s_coms_%s.npy' %(homedir,project,task,atlas,run_version),np.array(subject_communities)) 
		np.save('%sdynamic_mod/results/%s_%s_%s_z_matrices_%s.npy'%(homedir,project,task,atlas,run_version),np.array(thresh_matrices))
	subject_mods = np.array(subject_mods)
	subject_pcs = np.array(subject_pcs)
	subject_wmds = np.array(subject_wmds)
	subject_communities = np.array(subject_communities)
	matrices = np.array(matrices)
	thresh_matrices = np.array(thresh_matrices)
	results = {}
	results['subject_pcs'] = subject_pcs
	results['subject_mods'] = subject_mods
	results['subject_wmds'] = subject_wmds
	results['subject_communities'] = subject_communities
	results['matrices'] = matrices
	del matrices
	results['z_scored_matrices'] = thresh_matrices
	results['subjects'] = finished_subjects
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

def pc_edge_q_figure(tasks = ['REST','WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL']):
	"""
	edge weight mediation of pearsonr(PC,Q) = 
	(regression coefficient of edge weight by PC) How much variance in the edge is explained by PC
	(regression coefficient of Q by edge weight, controlling for PC) How much variance in Q is explained by the edge weight, controlling for PC.
	"""
	driver = 'PC'
	project='hcp'
	atlas = 'power'
	run_version = 'fz'
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	q_corr_matrix = []
	pc_corr_matrix = []
	#order by primary versus secondary networks. 
	network_order = ['Auditory','Sensory/somatomotor Hand','Sensory/somatomotor Mouth','Visual','Dorsal attention','Ventral attention',
	'Cingulo-opercular Task Control','Salience','Fronto-parietal Task Control','Default mode','Cerebellar','Subcortical','Memory retrieval?','Uncertain']
	colors = np.array(pd.read_csv('%smodularity/Consensus264.csv'%(homedir),header=None)[34].values)
	colors[colors=='Pale blue'] = '#ADD8E6'
	colors[colors=='Teal'] = '#008080'
	swap_indices = []
	for nn in network_order:
		original_idx = np.where(network_names == nn)
		for i in range(len(original_idx[0])):
			swap_indices.append(original_idx[0][i])
	locality_df = pd.DataFrame()
	stats = []
	for task in tasks:
		print task
		subjects = np.load('%sdynamic_mod/results/%s_%s_%s_subs_%s.npy' %(homedir,project,task,atlas,run_version))
		static_results = graph_metrics(subjects,task,atlas,run_version)
		subject_pcs = static_results['subject_pcs']
		subject_mods = static_results['subject_mods']
		subject_wmds = static_results['subject_wmds']
		matrices = static_results['matrices']
		task_perf = task_performance(subjects,task)
		scipy.stats.zscore(task_perf)
		assert subject_pcs.shape[0] == len(subjects)
		mean_pc = np.nanmean(subject_pcs,axis=0)
		mean_wmd = np.nanmean(subject_wmds,axis=0)
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
		mod_wmd_corr = np.zeros(subject_wmds.shape[1])
		for i in range(subject_wmds.shape[1]):
			mod_wmd_corr[i] = nan_pearsonr(subject_mods,subject_wmds[:,i])[0]
		m = np.load('%s/dynamic_mod/results/full_med_matrix_new_%s.npy'%(homedir,task))
		mean_conn = np.nanmean(matrices,axis=0)
		e_tresh = np.percentile(mean_conn,85)
		for i in range(264):
			real_t = scipy.stats.ttest_ind(np.abs(m)[i][np.argwhere(mean_conn[i]>=e_tresh)][:,:,np.arange(264)!=i].reshape(-1),np.abs(m)[i][np.argwhere(mean_conn[i]<e_tresh)][:,:,np.arange(264)!=i].reshape(-1))[0]
			# real_t = scipy.stats.ttest_ind(np.abs(m)[i][np.argsort(mean_conn[i])[-50:]][:,np.arange(264)!=i].reshape(-1),np.abs(m)[np.argsort(mean_conn[i])[-50:]][:,np.arange(264)!=i].reshape(-1))[0]
			if mod_pc_corr[i] > 0.0:
				locality_df = locality_df.append({"Node Type":'Connector Hub','t':real_t,'Task':task.capitalize()},ignore_index=True)
			else:
				locality_df = locality_df.append({"Node Type":'Local Node','t':real_t,'Task':task.capitalize()},ignore_index=True)
		locality_df.dropna(inplace=True)
		stat = tstatfunc(locality_df.t[(locality_df["Node Type"]=='Connector Hub')&(locality_df["Task"]==task.capitalize())],locality_df.t[(locality_df["Node Type"]=='Local Node')&(locality_df["Task"]==task.capitalize())])
		print stat
		stats.append(stat)
		if driver == 'PC':
			predict_nodes = np.where(mod_pc_corr>0.0)[0]
			local_predict_nodes = np.where(mod_pc_corr<0.0)[0]
			pc_edge_corr = np.arctanh(pc_edge_correlation(subject_pcs,matrices,path='%s/dynamic_mod/results/%s_%s_%s_pc_edge_corr_z.npy' %(homedir,project,task,atlas)))
		if driver == 'WMD':
			predict_nodes = np.where(mod_wmd_corr>0.0)[0]
			local_predict_nodes = np.where(mod_wmd_corr<0.0)[0]
			pc_edge_corr = np.arctanh(pc_edge_correlation(subject_wmds,matrices,path='%s/dynamic_mod/results/%s_%s_%s_wmd_edge_corr_z.npy' %(homedir,project,task,atlas)))
		n_nodes = pc_edge_corr.shape[0]
		q_edge_corr = np.zeros((n_nodes,n_nodes))
		perf_edge_corr = np.zeros((n_nodes,n_nodes))
		for i,j in combinations(range(n_nodes),2):
			ijqcorr = nan_pearsonr(matrices[:,i,j],subject_mods)[0]
			q_edge_corr[i,j] = ijqcorr
			q_edge_corr[j,i] = ijqcorr
			if task not in ['WM','RELATIONAL','SOCIAL','LANGUAGE']:
				continue
			ijqcorr = nan_pearsonr(matrices[:,i,j],task_perf)[0]
			perf_edge_corr[i,j] = ijqcorr
			perf_edge_corr[j,i] = ijqcorr
		pc_corr_matrix.append(np.nanmean(pc_edge_corr[predict_nodes,:,:],axis=0))
		q_corr_matrix.append(q_edge_corr)

		# plot_corr_matrix(np.nanmean(q_corr_matrix,axis=0)[:,swap_indices][swap_indices],network_names.copy()[swap_indices],out_file='%s/dynamic_mod/figures/%s_%s_edge_q_corr_matrix.pdf'%(homedir,tasks,run_version),block_lower=False,return_array=False,plot_corr=True,label=False,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True)
		print np.nanmin(pc_corr_matrix),np.nanmax(pc_corr_matrix)
		print np.nanmin(q_corr_matrix),np.nanmax(q_corr_matrix)
		print np.nanmin(np.nanmean(m[predict_nodes,:,:],axis=0)),np.nanmax(np.nanmean(m[predict_nodes,:,:],axis=0))
		# if task in ['WM','RELATIONAL','SOCIAL','LANGUAGE']:
			# print nan_pearsonr(perf_edge_corr.reshape(-1),np.nanmean(pc_edge_corr[predict_nodes,:,:],axis=0).reshape(-1))
			# plot_corr_matrix(perf_edge_corr[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_%s_edge_perf_corr_matrix.pdf'%(homedir,task,run_version),block_lower=False,return_array=False,plot_corr=True,label=False,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True)
		plot_corr_matrix(np.nanmean(m[predict_nodes,:,:],axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_%s_%s_mediation_matrix.pdf'%(homedir,task,driver,run_version),block_lower=False,return_array=False,plot_corr=True,label=False,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True)
		# plot_corr_matrix(np.nanmean(pc_corr_matrix,axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_%s_edge_q_corr_matrix.pdf'%(homedir,task,run_version),block_lower=False,return_array=False,plot_corr=True,label=False,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True)
		# plot_corr_matrix(np.nanmean(q_corr_matrix,axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_%s_%s_edgecorr_matrix.pdf'%(homedir,task,driver,run_version),block_lower=False,return_array=False,plot_corr=True,label=False,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True)
		
		# plot_corr_matrix(np.nanmean(m[predict_nodes,:,:],axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file=None,block_lower=False,return_array=False,plot_corr=True,label=False,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True)
	# plot_corr_matrix(np.nanmean(q_corr_matrix,axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_mean_q_corr_matrix.pdf'%(homedir,run_version),block_lower=False,return_array=False,plot_corr=True,label=False,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True)
	# plot_corr_matrix(np.nanmean(pc_corr_matrix,axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_mean_pc_corr_matrix.pdf'%(homedir,run_version),block_lower=False,return_array=False,plot_corr=True,label=False,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True)
	# plot_corr_matrix(np.nanmean(m[predict_nodes,:,:],axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_%s_%s_mediation_matrix_withbar.pdf'%(homedir,task,driver,run_version),block_lower=False,return_array=False,plot_corr=True,label=False,reorder=False,colors=colors[swap_indices],line=True,colorbar=True,draw_legend=True)	
	f = sns.plt.figure(figsize=(18,6))
	sns.set_style("white")
	sns.set_style("ticks")
	sns.set(context="paper",font='Helvetica',font_scale=1.2)
	# sns.violinplot(data=locality_df[locality_df['Node Type']=='Connector Hub'],x='Task',y='t',inner='quartile',palette=sns.palettes.color_palette('Paired',7))
	sns.violinplot(data=locality_df,y='t',hue="Node Type",x='Task',split=True,inner='quartile',palette=sns.palettes.color_palette('Paired',7))
	# for i,stat in enumerate(stats):
	# 	sns.plt.text(i,0.05,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.ylabel("T Test Values, mediation values of node's nieghbors \n versus mediation of node's non-neighbors")
	sns.plt.legend(bbox_to_anchor=[1,1.05],ncol=7,columnspacing=10)
	sns.plt.savefig('%s/dynamic_mod/figures/%s_mediation_t_test.pdf'%(homedir,run_version))

def network_labels(atlas):
	if atlas == 'gordon':
		name_dict = {}
		df = pd.read_excel('%sdynamic_mod/Parcels.xlsx'%(homedir))
		df.Community[df.Community=='None'] = 'Uncertain'
		for i,com in enumerate(np.unique(df.Community.values)):
			name_dict[com] = i
		known_membership = np.zeros((333))
		for i in range(333):
			known_membership[i] = name_dict[df.Community[i]]
		network_names = np.array(df.Community.values).astype(str)
	if atlas == 'power':
		known_membership = np.array(pd.read_csv('%smodularity/Consensus264.csv'%(homedir),header=None)[31].values)
		known_membership[known_membership==-1] = 0
		network_names = np.array(pd.read_csv('%smodularity/Consensus264.csv'%(homedir),header=None)[36].values)
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

def run_fd(subject,task):
	try:
		MP = np.load('/home/despoB/mb3152/dynamic_mod/motion_files/%s_%s.npy' %(subject,task))
	except:
		outfile = '/home/despoB/mb3152/dynamic_mod/motion_files/%s_%s.npy' %(subject,task)
		MP = np.loadtxt('/home/despoB/connectome-raw/%s/MNINonLinear/Results/%s/Movement_Regressors.txt'%(subject,task))
		FD = brain_graphs.compute_FD(MP[:,:6])
		FD = np.append(0,FD)
		np.save(outfile,FD)
		MP = np.load('/home/despoB/mb3152/dynamic_mod/motion_files/%s_%s.npy' %(subject,task))
	return MP

def get_sub_motion(subject,task):
	motion_files = glob.glob('/home/despoB/mb3152/dynamic_mod/motion_files/%s_*%s*' %(subject,task))
	if len(motion_files) == 0:
		smo = np.nan
	if len(motion_files) > 0:
		smo = []
		for m in motion_files:
			smo.append(np.nanmean(np.load(m)))
		smo = np.nanmean(smo)
	return smo

def all_motion(tasks,atlas='power'):
	everything = ['tfMRI_WM_RL','tfMRI_WM_LR','rfMRI_REST1_LR','rfMRI_REST2_LR','rfMRI_REST1_RL','rfMRI_REST2_RL','tfMRI_RELATIONAL_LR','tfMRI_RELATIONAL_RL','tfMRI_SOCIAL_RL','tfMRI_SOCIAL_LR','tfMRI_LANGUAGE_LR','tfMRI_LANGUAGE_RL','tfMRI_GAMBLING_RL','tfMRI_MOTOR_RL','tfMRI_GAMBLING_LR','tfMRI_MOTOR_LR']
	for task in everything:
		print task
		if 'REST' in task:
			subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',task.split('_')[1][:4],atlas))
		else:
			subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',task.split('_')[1],atlas))
		for subject in subjects:
			try:
				run_fd(subject,task)
			except:
				print subject,task

def connectivity_across_tasks(atlas='power',project='hcp',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='fz_wc',control_com=False,control_motion=False):
	results_df = pd.DataFrame(columns=['Task','Analysis','Statistic', 'Result'])
	for task in tasks:
		print task
		subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_%s.npy' %('hcp',task,atlas,run_version))
		static_results = graph_metrics(subjects,task,atlas,run_version=run_version)
		subject_pcs = static_results['subject_pcs']
		subject_mods = static_results['subject_mods']
		subject_wmds = static_results['subject_wmds']
		subject_communities = static_results['subject_communities']
		matrices = static_results['matrices']
		subjects = static_results['subjects']
		if control_motion == True:
			subject_motion = []
			for subject in subjects:
				subject_motion.append(get_sub_motion(subject,task))
			assert (np.isnan(subject_motion)==True).any() == False
			assert np.min(subject_motion) > 0.
		mean_pc = np.nanmean(static_results['subject_pcs'],axis=0)
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		cmod_pc_corr = np.zeros(subject_pcs.shape[1])
		com_pc_corr = np.zeros(subject_pcs.shape[1])
		if control_com == True and control_motion == True:
			model_vars = np.array([subject_motion,subject_communities]).transpose()
			r_mod = sm.GLM(subject_mods,sm.add_constant(model_vars)).fit()
			assert np.isclose(0.0,pearsonr(r_mod.resid_response,subject_motion)[0]) == True
			assert np.isclose(0.0,pearsonr(r_mod.resid_response,subject_communities)[0]) == True
			c_str = 'Motion and Number of Communities'
		if control_com == True and control_motion == False:
			r_mod = sm.GLM(subject_mods,sm.add_constant(subject_communities)).fit()
			assert np.isclose(0.0,pearsonr(r_mod.resid_response,subject_communities)[0]) == True
			c_str = 'Number of Communities'
		if control_com == False and control_motion == True:
			r_mod = sm.GLM(subject_mods,sm.add_constant(subject_motion)).fit()
			assert np.isclose(0.0,pearsonr(r_mod.resid_response,subject_motion)[0]) == True
			c_str = 'Motion'
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_pcs[:,i],subject_mods)[0]
			if control_com == True or control_motion == True:
				cmod_pc_corr[i] = nan_pearsonr(subject_pcs[:,i],r_mod.resid_response)[0]
		print np.nanmax(mod_pc_corr),np.nanmean(mod_pc_corr)
		if control_com == True or control_motion == True:
			print np.nanmax(cmod_pc_corr),np.nanmean(cmod_pc_corr)
		mod_pc_corr_result = pearsonr(mod_pc_corr,mean_pc)
		print 'r, Mean PC, r(PC,Q): ', np.round(mod_pc_corr_result[0],3), np.round(mod_pc_corr_result[1],10)
		for i in range(subject_pcs.shape[1]):
			com_pc_corr[i] = nan_pearsonr(subject_pcs[:,i],subject_communities)[0]		
		results_df = results_df.append({'Task':task,'Analysis': 'r, Mean PC, r(PC,Q)','Statistic':'R, P', 'Result':'%s, %s' %(np.round(mod_pc_corr_result[0],3),np.round(mod_pc_corr_result[1],10))},ignore_index=True)
		if control_com == True or control_motion == True:
			control_mod_pc_corr_result = pearsonr(cmod_pc_corr,mean_pc)
			print '%s Controlled r: ' %(c_str), np.round(control_mod_pc_corr_result[0],3), np.round(control_mod_pc_corr_result[1],10)
			results_df = results_df.append({'Task':task,'Analysis': 'r, Mean PC, r(PC,Q), %s Controlled r: ' %(c_str),'Statistic':'R, P', 'Result':'%s, %s' %(str(np.round(control_mod_pc_corr_result[0],3)), str(np.round(control_mod_pc_corr_result[1],10)))},ignore_index=True)
		if task in ['WM','RELATIONAL','LANGUAGE','SOCIAL']:
			task_perf = task_performance(subjects,task)
			to_delete = np.isnan(task_perf).copy()
			to_delete = np.where(to_delete==True)
			task_perf = np.delete(task_perf,to_delete)
			task_perf = scipy.stats.zscore(task_perf)
			subject_pcs = np.delete(subject_pcs,to_delete,axis=0)
			subject_communities = np.delete(subject_communities,to_delete)
			matrices = np.delete(matrices,to_delete,axis=0)
			subject_mods = np.delete(subject_mods,to_delete)
			subject_wmds = np.delete(subject_wmds,to_delete,axis=0)
			subject_pcs[np.isnan(subject_pcs)] = 0.0
			subject_wmds[np.isnan(subject_wmds)] = 0.0
			if control_motion == True:
				subject_motion = np.delete(subject_motion,to_delete)
			mean_pc = np.nanmean(subject_pcs,axis=0)
			cmod_perf_corr = np.zeros(subject_pcs.shape[1])
			mod_perf_corr = np.zeros(subject_pcs.shape[1])
			if control_com == True and control_motion == True:
				model_vars = np.array([subject_motion,subject_communities]).transpose()
				r_perf = sm.GLM(task_perf,sm.add_constant(model_vars)).fit()
				assert np.isclose(0.0,pearsonr(r_perf.resid_response,subject_motion)[0]) == True
				assert np.isclose(0.0,pearsonr(r_perf.resid_response,subject_communities)[0]) == True
			if control_com == True and control_motion == False:
				r_perf = sm.GLM(task_perf,sm.add_constant(subject_communities)).fit()
				assert np.isclose(0.0,pearsonr(r_perf.resid_response,subject_communities)[0]) == True
			if control_com == False and control_motion == True:
				r_perf = sm.GLM(task_perf,sm.add_constant(subject_motion)).fit()
				assert np.isclose(0.0,pearsonr(r_perf.resid_response,subject_motion)[0]) == True
			for i in range(subject_pcs.shape[1]):
				mod_perf_corr[i] = nan_pearsonr(task_perf,subject_pcs[:,i])[0]
				if control_com == True or control_motion == True:
					cmod_perf_corr[i] = nan_pearsonr(r_perf.resid_response,subject_pcs[:,i])[0]
			print np.nanmax(mod_perf_corr),np.nanmean(mod_perf_corr)
			if control_com == True or control_motion == True:
				print np.nanmax(cmod_perf_corr),np.nanmean(cmod_perf_corr)
			result = pearsonr(mod_perf_corr,mean_pc)
			cresult = pearsonr(cmod_perf_corr,mean_pc)
			print 'r, Mean PC, r(PC,Performance): ', np.round(result[0],3), np.round(result[1],10)
			results_df = results_df.append({'Task':task,'Analysis': 'r, Mean PC, r(PC,Performance): ','Statistic':'R, P', 'Result':'%s, %s' %(str(np.round(result[0],3)), str(np.round(result[1],10)))},ignore_index=True)
			if control_com == True or control_motion == True:
				print '%s Controlled r: ' %(c_str), np.round(cresult[0],3), np.round(cresult[1],10)
				results_df = results_df.append({'Task':task,'Analysis': 'r, Mean PC, r(PC,Performance), %s Controlled r:  '%(c_str),'Statistic':'R, P', 'Result':'%s, %s' %(str(np.round(cresult[0],3)), str(np.round(cresult[1],10)))},ignore_index=True)
	return results_df

def connectivity_across_tasks_old(atlas='power',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']):
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
		all_matrices = []
		violin_df = pd.DataFrame()
		for task in tasks:
			# subjects = np.array(hcp_subjects).copy()
			# subjects = list(subjects)
			# subjects = remove_missing_subjects(subjects,task,atlas)
			subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',task,atlas))
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
			if driver == 'PC':
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
			if driver == 'PC':
				print task + ', Connector Hubs(Q+): ' + str(scipy.stats.ttest_ind(task_violin_df["r value, node i's PCs and j's edge weights"][task_violin_df['Node Type']=='Q+'][task_violin_df['Edge Type']=='Within Community'],
					task_violin_df["r value, node i's PCs and j's edge weights"][task_violin_df['Node Type']=='Q+'][task_violin_df['Edge Type']=='Between Community']))
				print task + ', Non-Connector Hubs(Q-): ' + str(scipy.stats.ttest_ind(task_violin_df["r value, node i's PCs and j's edge weights"][task_violin_df['Node Type']=='Q-'][task_violin_df['Edge Type']=='Within Community'],
					task_violin_df["r value, node i's PCs and j's edge weights"][task_violin_df['Node Type']=='Q-'][task_violin_df['Edge Type']=='Between Community']))
			else:
				print task + ', Local Hubs(Q+): ' + str(scipy.stats.ttest_ind(task_violin_df["r value, node i's PCs and j's edge weights"][task_violin_df['Node Type']=='Q+'][task_violin_df['Edge Type']=='Within Community'],
					task_violin_df["r value, node i's PCs and j's edge weights"][task_violin_df['Node Type']=='Q+'][task_violin_df['Edge Type']=='Between Community']))
				print task + ', Non Local Hubs (Q-): ' + str(scipy.stats.ttest_ind(task_violin_df["r value, node i's PCs and j's edge weights"][task_violin_df['Node Type']=='Q-'][task_violin_df['Edge Type']=='Within Community'],
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
		if driver == 'PC':
			print task + ',Connector Hubs(Q+): ' + str(scipy.stats.ttest_ind(violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q+'][violin_df['Edge Type']=='Within Community'],
				violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q+'][violin_df['Edge Type']=='Between Community']))
			print task + ', Non-Connector Hubs(Q-): ' + str(scipy.stats.ttest_ind(violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q-'][violin_df['Edge Type']=='Within Community'],
				violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q-'][violin_df['Edge Type']=='Between Community']))
		else:
			print task + ', Local Hubs(Q+): ' + str(scipy.stats.ttest_ind(violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q+'][violin_df['Edge Type']=='Within Community'],
				violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q+'][violin_df['Edge Type']=='Between Community']))
			print task + ', Non-Local Hubs(Q-): ' + str(scipy.stats.ttest_ind(violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q-'][violin_df['Edge Type']=='Within Community'],
				violin_df["r value, node i's PCs and j's edge weights"][violin_df['Node Type']=='Q-'][violin_df['Edge Type']=='Between Community']))
		sns.set_style("white")
		sns.set_style("ticks")
		colors = sns.color_palette(['#fdfd96','#C4D8E2'])
		with sns.plotting_context("paper",font_scale=3):
			plt.figure(figsize=(24,16))
			sns.boxplot(x="Node Type", y="r value, node i's PCs and j's edge weights",hue="Edge Type", palette=colors,order=['Q+','Q-'], data=violin_df)
			plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/%s_edge_mod_avg.pdf'%(driver),dpi=4600)
			plt.close()

def individual_pc_q():
	atlas = 'power'
	project='hcp'
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	task = 'REST'
	print task
	subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',task,atlas))
	static_results = graph_metrics(subjects,task,atlas)
	matrices = static_results['matrices']
	subject_pcs = static_results['subject_pcs']
	subject_mods = static_results['subject_mods']
	mod_pc_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
	threshes = [0.1,.2,0.25,.3]
	threshes = [.1]
	mean_pc = np.nanmean(subject_pcs,axis=0)
	for thresh in threshes:
		df = pd.DataFrame()
		for i in range(len(subject_pcs)):
			for pi,p in enumerate(subject_pcs[i]):
				if mod_pc_corr[pi] < thresh:
					continue
				df = df.append({'node':pi,'Q':subject_mods[i],'PC':p},ignore_index=True)
				# df = df.append({'node':pi,'Performance':task_perf[i],'PC':p},ignore_index=True)
		sns.lmplot('PC','Q',df,hue='node',order=2,truncate=True,scatter=False,scatter_kws={'label':'Order:1','color':'y'})
		sns.plt.xlim([0,.75])
		sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/pc_mod_regress_%s_2.pdf'%(thresh))
		sns.plt.close()
		sns.lmplot('PC','Q',df,hue='node',order=3,truncate=True,scatter=False,scatter_kws={'label':'Order:1','color':'y'})
		sns.plt.xlim([0,.75])
		sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/pc_mod_regress_%s_3.pdf'%(thresh))
		sns.plt.close()
		sns.lmplot('PC','Q',df,hue='node',lowess=True,truncate=True,scatter=False,scatter_kws={'label':'Order:1','color':'y'})
		sns.plt.xlim([0,.75])
		sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/pc_mod_regress_%s_lowless.pdf'%(thresh))
		sns.plt.close()
		sns.lmplot('PC','Q',df,order=2,truncate=True,scatter=False,scatter_kws={'label':'Order:1','color':'y'})
		sns.plt.xlim([0,.75])
		sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/pc_mod_regress_mean_%s_2.pdf'%(thresh))
		sns.plt.close()
		sns.lmplot('PC','Q',df,order=3,truncate=True,scatter=False,scatter_kws={'label':'Order:1','color':'y'})
		sns.plt.xlim([0,.75])
		sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/pc_mod_regress_mean_%s_3.pdf'%(thresh))
		sns.plt.close()
		sns.lmplot('PC','Q',df,lowess=True,truncate=True,scatter=False,scatter_kws={'label':'Order:1','color':'y'})
		sns.plt.xlim([0,.75])
		sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/pc_mod_regress_mean_%s_lowless.pdf'%(thresh))
		sns.plt.close()

def multi_med(data):
	outcome_model = sm.OLS.from_formula("q ~ weight + pc", data)
	mediator_model = sm.OLS.from_formula("weight ~ pc", data)
	med_val = np.mean(Mediation(outcome_model, mediator_model, "pc", "weight").fit(n_rep=10).ACME_avg)
	return med_val

def mediation(task):
	"""
	264,264,264 matrix, which edges mediate the relationship between PC and Q
	"""
	atlas = 'power'
	project='hcp'
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	subjects = np.load('%s/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %(homedir,'hcp',task,atlas))
	static_results = graph_metrics(subjects,task,atlas,run_version='fz')
	matrices = static_results['matrices']
	subject_pcs = static_results['subject_pcs']
	subject_mods = static_results['subject_mods']
	mod_pc_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
	mean_conn = np.nanmean(matrices,axis=0)
	e_tresh = np.percentile(mean_conn,85)
	try:
		m = np.load('%s/dynamic_mod/results/full_med_matrix_new_%s.npy'%(homedir,task))
	except:
		subject_pcs[np.isnan(subject_pcs)] = 0.0
		m = np.zeros((264,264,264))
		pool = Pool(40)
		for n in range(264):
			print n
			sys.stdout.flush()
			variables =  []
			for i,j in combinations(range(264),2):
				variables.append(pd.DataFrame(data={'pc':subject_pcs[:,n],'weight':matrices[:,i,j],'q':subject_mods},index=range(len(subject_pcs))))
			results = pool.map(multi_med,variables)
			for r,i in zip(results,combinations(range(264),2)):
				m[n,i[0],i[1]] = r
				m[n,i[1],i[0]] = r
			np.save('/home/despoB/mb3152/dynamic_mod/results/full_med_matrix_new_%s.npy'%(task),m)
	locality_df = pd.DataFrame()
	for i in range(264):
		real_t = scipy.stats.ttest_ind(np.abs(m[i][np.argwhere(mean_conn[i]>e_tresh)][:,:,np.arange(264)!=i].reshape(-1)),np.abs(m[i][np.argwhere(mean_conn[i]<e_tresh)][:,:,np.arange(264)!=i].reshape(-1)))[0]
		if mod_pc_corr[i] > 0.0:
			locality_df = locality_df.append({"Nodes' Correlation with Q":'+','t':real_t},ignore_index=True)
		else:
			locality_df = locality_df.append({"Nodes' Correlation with Q":'-','t':real_t},ignore_index=True)
	locality_df.dropna(inplace=True)
	stat = tstatfunc(locality_df.t[locality_df["Nodes' Correlation with Q"]=='+'],locality_df.t[locality_df["Nodes' Correlation with Q"]=='-'])
	print stat
	sns.set_style("white")
	sns.set_style("ticks")
	sns.violinplot(data=locality_df,y='t',x="Nodes' Correlation with Q",order = ['+','-'],inner='quartile',palette=np.array(sns.color_palette("cubehelix", 10))[[7,9]])
	sns.plt.text(.5,50,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.ylabel('T Test Values, mediation values of nieghbors versus mediation of non-neighbors')
	sns.plt.savefig('%s/dynamic_mod/figures/mediation_t_test_%s.pdf'%(homedir,task))
	sns.plt.show()

def local_mediation(task):
	"""
	264,264,264 matrix, which edges mediate the relationship between PC and Q
	"""
	atlas = 'power'
	project='hcp'
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	subjects = np.load('%s/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %(homedir,'hcp',task,atlas))
	static_results = graph_metrics(subjects,task,atlas,run_version='fz')
	matrices = static_results['matrices']
	subject_pcs = static_results['subject_pcs']
	subject_wmds = static_results['subject_wmds']
	subject_mods = static_results['subject_mods']
	mod_wmd_corr = np.zeros(subject_wmds.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_wmd_corr[i] = nan_pearsonr(subject_mods,subject_wmds[:,i])[0]
	mean_conn = np.nanmean(matrices,axis=0)
	e_tresh = np.percentile(mean_conn,85)
	try:
		m = np.load('%s/dynamic_mod/results/full_med_matrix_new_%s_wmds.npy'%(homedir,task))
	except:
		subject_wmds[np.isnan(subject_pcs)] = 0.0
		m = np.zeros((264,264,264))
		pool = Pool(40)
		for n in range(264):
			print n
			sys.stdout.flush()
			variables =  []
			for i,j in combinations(range(264),2):
				variables.append(pd.DataFrame(data={'pc':subject_wmds[:,n],'weight':matrices[:,i,j],'q':subject_mods},index=range(len(subject_pcs))))
			results = pool.map(multi_med,variables)
			for r,i in zip(results,combinations(range(264),2)):
				m[n,i[0],i[1]] = r
				m[n,i[1],i[0]] = r
			np.save('/home/despoB/mb3152/dynamic_mod/results/full_med_matrix_new_%s_wmds.npy'%(task),m)
	locality_df = pd.DataFrame()
	for i in range(264):
		real_t = scipy.stats.ttest_ind(np.abs(m[i][np.argwhere(mean_conn[i]>e_tresh)][:,:,np.arange(264)!=i].reshape(-1)),np.abs(m[i][np.argwhere(mean_conn[i]<e_tresh)][:,:,np.arange(264)!=i].reshape(-1)))[0]
		if mod_wmd_corr[i] > 0.0:
			locality_df = locality_df.append({"Nodes' Correlation with Q":'+','t':real_t},ignore_index=True)
		else:
			locality_df = locality_df.append({"Nodes' Correlation with Q":'-','t':real_t},ignore_index=True)
	locality_df.dropna(inplace=True)
	stat = tstatfunc(locality_df.t[locality_df["Nodes' Correlation with Q"]=='+'],locality_df.t[locality_df["Nodes' Correlation with Q"]=='-'])
	print stat
	# sns.set_style("white")
	# sns.set_style("ticks")
	# sns.violinplot(data=locality_df,y='t',x="Nodes' Correlation with Q",order = ['+','-'],inner='quartile',palette=np.array(sns.color_palette("cubehelix", 10))[[7,9]])
	# sns.plt.text(.5,50,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	# sns.plt.ylabel('T Test Values, mediation values of nieghbors versus mediation of non-neighbors')
	# sns.plt.savefig('%s/dynamic_mod/figures/mediation_t_test_%s.pdf'%(homedir,task))
	# sns.plt.show()

def local_versus_connector_mediation(task):
	task = 'WM'
	atlas = 'power'
	project='hcp'
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	subjects = np.load('%s/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %(homedir,'hcp',task,atlas))
	static_results = graph_metrics(subjects,task,atlas,run_version='fz')
	matrices = static_results['matrices']
	subject_pcs = static_results['subject_pcs']
	subject_wmds = static_results['subject_wmds']
	subject_mods = static_results['subject_mods']
	mod_wmd_corr = np.zeros(subject_wmds.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_wmd_corr[i] = nan_pearsonr(subject_mods,subject_wmds[:,i])[0]
	mod_pc_corr = np.zeros(subject_wmds.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
	mean_conn = np.nanmean(matrices,axis=0)
	e_tresh = np.percentile(mean_conn,50)
	local = np.load('%s/dynamic_mod/results/full_med_matrix_new_%s_wmds.npy'%(homedir,task))
	connector = np.load('%s/dynamic_mod/results/full_med_matrix_new_%s_wmds.npy'%(homedir,task))
	local = np.abs(local)
	connector = np.abs(connector)
	locality_df = pd.DataFrame()
	for i in range(132):
		if i in np.argsort(mod_wmd_corr)[-100:]:
			real_t = scipy.stats.ttest_ind(local[i][np.argwhere(mean_conn[i]>e_tresh)][:,:,np.arange(264)!=i].reshape(-1),local[i][np.argwhere(mean_conn[i]<e_tresh)][:,:,np.arange(264)!=i].reshape(-1))[0]
			locality_df = locality_df.append({"Node Type":'Local Hub','t':real_t},ignore_index=True)
		if i in np.argsort(mod_pc_corr)[-100:]:
			real_t = scipy.stats.ttest_ind(connector[i][np.argwhere(mean_conn[i]>e_tresh)][:,:,np.arange(264)!=i].reshape(-1),connector[i][np.argwhere(mean_conn[i]<e_tresh)][:,:,np.arange(264)!=i].reshape(-1))[0]
			locality_df = locality_df.append({"Node Type":'Connector Hub','t':real_t},ignore_index=True)
	locality_df.dropna(inplace=True)
	stat = tstatfunc(locality_df.t[locality_df["Node Type"]=='Connector Hub'],locality_df.t[locality_df["Node Type"]=='Local Hub'])
	print stat

def sm_null():
	try:
		r = np.load('/home/despoB/mb3152/dynamic_mod/results/null_sw_results.npy')
	except:
		sw_rs = []
		sw_crs = []
		for i in range(100):
			print i
			pc = []
			mod = []
			wmd = []
			memlen = []
			for s in range(100):
				graph = Graph.Watts_Strogatz(1,264,7,.25)
				graph.es["weight"] = np.ones(graph.ecount())
				graph = graph.community_infomap()
				graph = brain_graphs.brain_graph(graph)
				pc.append(np.array(graph.pc))
				wmd.append(np.array(graph.wmd))
				mod.append(graph.community.modularity)
				memlen.append(len(graph.community.sizes()))
			pc = np.array(pc)
			mod = np.array(mod)
			wmd = np.array(wmd)
			memlen = np.array(memlen)
			mod_pc_corr = np.zeros(264)
			for i in range(264):
				mod_pc_corr[i] = nan_pearsonr(mod,pc[:,i])[0]
			print pearsonr(np.nanmean(pc,axis=0),mod_pc_corr)[0]
			print pearsonr(mod,memlen)[0]
			sw_rs.append(pearsonr(np.nanmean(pc,axis=0),mod_pc_corr)[0])
			sw_crs.append(pearsonr(mod,memlen)[0])
		r = np.array([sw_rs,sw_crs])
		np.save('/home/despoB/mb3152/dynamic_mod/results/null_sw_results.npy',r)
	return r

def null():
	sm_null_results = sm_null()[0]
	atlas = 'power'
	project='hcp'
	task = 'REST'
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',task,atlas))
	static_results = graph_metrics(subjects,task,atlas,'fz')
	subject_pcs = static_results['subject_pcs']
	subject_wmds = static_results['subject_wmds']
	subject_mods = static_results['subject_mods']
	subject_wmds = static_results['subject_wmds']
	matrices = static_results['matrices']
	try:
		null_graph_rs,null_community_rs,null_all_rs = np.load('/home/despoB/mb3152/dynamic_mod/results/null_results.npy')
		real_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/real_real_results.csv')
	except:
		1/0
		null_graph_rs = []
		null_community_rs = []
		null_all_rs = []
		for i in range(100):
			pool = Pool(40)
			n_g = pool.map(null_graph_individual_graph_analyes,matrices)
			n_c = pool.map(null_community_individual_graph_analyes,matrices)
			n_a = pool.map(null_all_individual_graph_analyes,matrices)
			"""
			null graph
			"""
			n_g_subject_pcs = []
			n_g_subject_wmds = []
			n_g_subject_mods = []
			for mod,pc,wmd in n_g:
				n_g_subject_mods.append(mod)
				n_g_subject_pcs.append(pc)
				n_g_subject_wmds.append(wmd)
			n_g_subject_pcs = np.array(n_g_subject_pcs)
			n_g_subject_wmds = np.array(n_g_subject_wmds)
			n_g_subject_mods = np.array(n_g_subject_mods)
			mean_pc = np.nanmean(n_g_subject_pcs,axis=0)
			mean_wmd = np.nanmean(n_g_subject_wmds,axis=0)
			n_g_mod_pc_corr = np.zeros(n_g_subject_pcs.shape[1])
			for i in range(n_g_subject_pcs.shape[1]):
				n_g_mod_pc_corr[i] = nan_pearsonr(n_g_subject_mods,n_g_subject_pcs[:,i])[0]
			n_g_mod_wmd_corr = np.zeros(n_g_subject_wmds.shape[1])
			for i in range(n_g_subject_wmds.shape[1]):
				n_g_mod_wmd_corr[i] = nan_pearsonr(n_g_subject_mods,n_g_subject_wmds[:,i])[0]
			print 'Pearson R, PC & Q, Mean PC: ', nan_pearsonr(n_g_mod_pc_corr,mean_pc)
			null_graph_rs.append(nan_pearsonr(n_g_mod_pc_corr,mean_pc)[0])
			# print 'Pearson R, PC & WCD, Mean WMD: ', nan_pearsonr(n_g_mod_wmd_corr,mean_wmd)

			n_c_subject_pcs = []
			n_c_subject_wmds = []
			n_c_subject_mods = []
			for mod,pc,wmd in n_c:
				n_c_subject_mods.append(mod)
				n_c_subject_pcs.append(pc)
				n_c_subject_wmds.append(wmd)
			n_c_subject_pcs = np.array(n_c_subject_pcs)
			n_c_subject_wmds = np.array(n_c_subject_wmds)
			n_c_subject_mods = np.array(n_c_subject_mods)
			mean_pc = np.nanmean(n_c_subject_pcs,axis=0)
			mean_wmd = np.nanmean(n_c_subject_wmds,axis=0)
			n_c_mod_pc_corr = np.zeros(n_c_subject_pcs.shape[1])
			for i in range(n_c_subject_pcs.shape[1]):
				n_c_mod_pc_corr[i] = nan_pearsonr(n_c_subject_mods,n_c_subject_pcs[:,i])[0]
			n_c_mod_wmd_corr = np.zeros(n_c_subject_wmds.shape[1])
			for i in range(n_c_subject_wmds.shape[1]):
				n_c_mod_wmd_corr[i] = nan_pearsonr(n_c_subject_mods,n_c_subject_wmds[:,i])[0]
			print 'Pearson R, PC & Q, Mean PC: ', nan_pearsonr(n_c_mod_pc_corr,mean_pc)
			null_community_rs.append(nan_pearsonr(n_c_mod_pc_corr,mean_pc)[0])
			# print 'Pearson R, PC & WCD, Mean WMD: ', nan_pearsonr(mod_wmd_corr,mean_wmd)

			n_a_subject_pcs = []
			n_a_subject_wmds = []
			n_a_subject_mods = []
			for mod,pc,wmd in n_a:
				n_a_subject_mods.append(mod)
				n_a_subject_pcs.append(pc)
				n_a_subject_wmds.append(wmd)
			n_a_subject_pcs = np.array(n_a_subject_pcs)
			n_a_subject_wmds = np.array(n_a_subject_wmds)
			n_a_subject_mods = np.array(n_a_subject_mods)
			mean_pc = np.nanmean(n_a_subject_pcs,axis=0)
			mean_wmd = np.nanmean(n_a_subject_wmds,axis=0)
			n_a_mod_pc_corr = np.zeros(n_a_subject_pcs.shape[1])
			for i in range(n_a_subject_pcs.shape[1]):
				n_a_mod_pc_corr[i] = nan_pearsonr(n_a_subject_mods,n_a_subject_pcs[:,i])[0]
			n_a_mod_wmd_corr = np.zeros(n_a_subject_wmds.shape[1])
			for i in range(n_a_subject_wmds.shape[1]):
				n_a_mod_wmd_corr[i] = nan_pearsonr(n_a_subject_mods,n_a_subject_wmds[:,i])[0]
			print 'Pearson R, PC & Q, Mean PC: ', nan_pearsonr(n_a_mod_pc_corr,mean_pc)
			null_all_rs.append(nan_pearsonr(n_a_mod_pc_corr,mean_pc)[0])
			# print 'Pearson R, PC & WCD, Mean WMD: ', nan_pearsonr(mod_n_a_wmd_corr,mean_wmd)
		results = np.array([null_graph_rs,null_community_rs,null_all_rs])
		np.save('/home/despoB/mb3152/dynamic_mod/results/null_results.npy',results)
		real_df = connectivity_across_tasks(atlas='power',project='hcp',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='fz_wc',control_com=False,control_motion=False)
		real_df.to_csv('/home/despoB/mb3152/dynamic_mod/results/real_real_results.csv')
	df = pd.DataFrame(columns=['R','Null Model Type'])
	for r in null_graph_rs:
		df = df.append({'R':r,'Null Model Type':'Random Edges, Real Community'},ignore_index=True)
	for r in null_community_rs:
		df = df.append({'R':r,'Null Model Type':'Random Community, Real Edges'},ignore_index=True)
	for r in null_all_rs:
		df = df.append({'R':r,'Null Model Type':'Random Edges, Clustered'},ignore_index=True)
	for r in sm_null_results:
		df = df.append({'R':r,'Null Model Type':'Wattz-Strogatz'},ignore_index=True)
	for r in real_df.Result:
		r = float(r.split(',')[0])
		df = df.append({'R':r,'Null Model Type':'Real Edges, Real Community'},ignore_index=True)
	f = sns.plt.figure(figsize=(18,8))
	sns.set_style("white")
	sns.set_style("ticks")
	sns.set(context="paper",font='Helvetica',font_scale=1.75)	
	sns.violinplot(x="Null Model Type", y="R", data=df,inner='quartile')
	sns.plt.ylabel("R Values Between Nodes' Mean Participation Coefficients\n and the R values of Participation Coefficients and Qs")
	sns.plt.tight_layout()
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/null_models.pdf')

def specificity():
	"""
	Specificity of modulation by nodes' PC.
	Does the PC value of i impact the connectivity of j as i and j are more strongly connected?
	"""
	atlas = 'power'
	project='hcp'
	df_columns=['Task','Hub Measure','Q+/Q-','Average Edge i-j Weight',"Strength of r's, i's PC & j's Q"]
	tasks = ['REST','WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL',]
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	df = pd.DataFrame(columns = df_columns)
	for task in tasks:
		print task
		# subjects = np.array(hcp_subjects).copy()
		# subjects = list(subjects)
		# subjects = remove_missing_subjects(subjects,task,atlas)
		subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',task,atlas))
		static_results = graph_metrics(subjects,task,atlas,'fz')
		subject_pcs = static_results['subject_pcs']
		subject_wmds = static_results['subject_wmds']
		subject_mods = static_results['subject_mods']
		subject_wmds = static_results['subject_wmds']
		matrices = static_results['matrices']
		#sum of weight changes for each node, by each node.
		hub_nodes = ['PC','WCD']
		# hub_nodes = ['PC']
		driver_nodes_list = ['Q+','Q-']
		# driver_nodes_list = ['Q+']
		mean_pc = np.nanmean(subject_pcs,axis=0)
		mean_wmd = np.nanmean(subject_wmds,axis=0)
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
		mod_wmd_corr = np.zeros(subject_wmds.shape[1])
		for i in range(subject_wmds.shape[1]):
			mod_wmd_corr[i] = nan_pearsonr(subject_mods,subject_wmds[:,i])[0]
		for hub_node in hub_nodes:
			if hub_node == 'PC':
				pc_edge_corr = np.arctanh(pc_edge_correlation(subject_pcs,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_pc_edge_corr_z.npy' %(project,task,atlas)))
				connector_nodes = np.where(mod_pc_corr>0.0)[0]
				local_nodes = np.where(mod_pc_corr<0.0)[0]
			else:
				pc_edge_corr = np.arctanh(pc_edge_correlation(subject_wmds,matrices,path='/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_wmd_edge_corr_z.npy' %(project,task,atlas)))
				connector_nodes = np.where(mod_wmd_corr>0.0)[0]
				local_nodes = np.where(mod_wmd_corr<0.0)[0]
			edge_thresh_val = 50.0
			edge_thresh = np.percentile(np.nanmean(matrices,axis=0),edge_thresh_val)
			pc_edge_corr[:,np.nanmean(matrices,axis=0)<edge_thresh] = np.nan
			for driver_nodes in driver_nodes_list:
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
					array = pc_edge_corr[n1][n2]
					weight_change_matrix_between[n1,n2] = np.nansum(pc_edge_corr[n1][n2][np.where((known_membership!=known_membership[n2])&(range(264)!=n1))])
					weight_change_matrix_within[n1,n2] = np.nansum(pc_edge_corr[n1][n2][np.where((known_membership==known_membership[n2])&(range(264)!=n1))])
					# for n3 in range(264):
					# 	if n1 == n3:
					# 		continue
					# 	if known_membership[n3]!= known_membership[n2]:
					# 		weight_change_matrix_between[n1,n2] = np.nansum([weight_change_matrix_between[n1,n2],array[n3]])
					# 		between_len = between_len + 1
					# 	else:
					# 		weight_change_matrix_within[n1,n2] = np.nansum([weight_change_matrix_within[n1,n2],array[n3]])
					# 		community_len = community_len + 1
					# weight_change_matrix_within[n1,n2] = weight_change_matrix_within[n1,n2] / community_len
					# weight_change_matrix_between[n1,n2] = weight_change_matrix_between[n1,n2] / between_len
				temp_matrix = np.nanmean(matrices,axis=0)
				weight_matrix = weight_change_matrix_within-weight_change_matrix_between
				weight_matrix[np.isnan(weight_matrix)] = 0.0
				if hub_node == 'PC':
					df_columns=['Task','Hub Measure','Q+/Q-','Average Edge i-j Weight',"Strength of r's, i's PC & j's Q"]
				else:
					df_columns=['Task','Hub Measure','Q+/Q-','Average Edge i-j Weight',"Strength of r's, i's WCD & j's Q"]
				df_array = []
				for i,j in zip(temp_matrix[weight_matrix!=0.0].reshape(-1),weight_matrix[weight_matrix!=0.0].reshape(-1)):
					df_array.append([task,hub_node,driver_nodes,i,j])
				df = pd.concat([df,pd.DataFrame(df_array,columns=df_columns)],axis=0)
				print hub_node, driver_nodes
				print pearsonr(weight_matrix[weight_matrix!=0.0].reshape(-1),temp_matrix[weight_matrix!=0.0].reshape(-1))

	# plot_connectivity_results(df[(df['Q+/Q-']=='Q+') &(df['Hub Measure']=='PC')],"Strength of r's, i's PC & j's Q",'Average Edge i-j Weight','/home/despoB/mb3152/dynamic_mod/figures/edge_spec_pcqplus_%s.pdf'%(edge_thresh_val))
	# plot_connectivity_results(df[(df['Q+/Q-']=='Q-') &(df['Hub Measure']=='PC')],"Strength of r's, i's PC & j's Q",'Average Edge i-j Weight','/home/despoB/mb3152/dynamic_mod/figures/edge_spec_pcqminus_%s.pdf'%(edge_thresh_val))
	plot_connectivity_results(df[(df['Q+/Q-']=='Q+') &(df['Hub Measure']=='WCD')],"Strength of r's, i's WCD & j's Q",'Average Edge i-j Weight','/home/despoB/mb3152/dynamic_mod/figures/edge_spec_wmdqplus_%s.pdf'%(edge_thresh_val))
	plot_connectivity_results(df[(df['Q+/Q-']=='Q-') &(df['Hub Measure']=='WCD')],"Strength of r's, i's WCD & j's Q",'Average Edge i-j Weight','/home/despoB/mb3152/dynamic_mod/figures/edge_spec_wmdqminus_%s.pdf'%(edge_thresh_val))
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

def between_community_centrality(graph,vc=None):
	if vc == None:
		vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	rank = graph.vcount()/5
	pc = vc.pc
	pc[np.isnan(pc)] = 0.0
	deg = np.array(vc.community.graph.strength(weights='weight'))
	return [np.array(graph.betweenness(weights='weight'))[np.argsort(pc)[-rank:]],
	np.array(graph.betweenness(weights='weight'))[np.argsort(deg)[-rank:]]]

def attack(variables):
	graph = variables[1]
	np.random.seed(variables[0])
	try: vc = variables[2]
	except: vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	pc = vc.pc
	pc[np.isnan(pc)] = 0.0
	deg = np.array(vc.community.graph.strength(weights='weight'))
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
	dmin = int(de*.5)
	dmax = int(de*.9)
	attack_degree_sps = []
	attack_pc_sps = []
	attack_degree_mods = []
	attack_pc_mods = []
	while True:
		num_kill = np.random.choice(range(dmin,dmax),1)
		np.random.shuffle(degree_edges)
		np.random.shuffle(connector_edges)
		d_edges = degree_edges[:num_kill]
		c_edges = connector_edges[:num_kill]
		d_graph = graph.copy()
		c_graph = graph.copy()
		for cnodes,dnodes in zip(c_edges.reshape(c_edges.shape[0],2),d_edges.reshape(d_edges.shape[0],2)):
			cnode1,cnode2,dnode1,dnode2  = cnodes[0],cnodes[1],dnodes[0],dnodes[1]
			d_graph.delete_edges([(dnode1,dnode2)])
			c_graph.delete_edges([(cnode1,cnode2)])
			if d_graph.is_connected() == False or c_graph.is_connected() == False:
				d_graph.add_edges([(dnode1,dnode2)])
				c_graph.add_edges([(cnode1,cnode2)])
		# delete_edges = []
		# for node1,node2 in d_edges.reshape(d_edges.shape[0],2):
		# 	delete_edges.append(graph.get_eid(node1,node2))
		# d_graph.delete_edges(delete_edges)
		# if d_graph.is_connected() == False:
		# 	continue
		# delete_edges = []
		# for node1,node2 in c_edges.reshape(c_edges.shape[0],2):
		# 	delete_edges.append(graph.get_eid(node1,node2))
		# c_graph.delete_edges(delete_edges)
		# if c_graph.is_connected() == False:
		# 	continue
		deg_sp = np.array(d_graph.shortest_paths()).astype(float)
		c_sp = np.array(c_graph.shortest_paths()).astype(float)
		attack_degree_sps.append(np.nansum(deg_sp))
		attack_pc_sps.append(np.nansum(c_sp))
		# attack_degree_mods.append(d_graph.community_infomap(edge_weights='weight').modularity)
		# attack_pc_mods.append(c_graph.community_infomap(edge_weights='weight').modularity)
		idx = idx + 1
		if idx == 500:
			break
	# return [attack_pc_sps,attack_degree_sps,healthy_sp,attack_pc_mods,attack_degree_mods,np.mean(healthy_mods)]
	return [attack_pc_sps,attack_degree_sps,healthy_sp]

def human_attacks():
	tasks = ['REST','WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL']
	try:
		df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/human_attack')
		btw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/human_bwt')
	except:
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
			cost = .2
			temp_matrix = np.nanmean(static_results['matrices'],axis=0).copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=cost,mst=True)
			for i in np.arange(20):
				variables.append([i,graph.copy(),brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))])
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
			variables = []
			for i in np.arange(20):
				cost = .2
				temp_matrix = np.nanmean(static_results['matrices'],axis=0).copy()
				graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=cost,mst=True)
				variables.append(graph)
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
	for task in tasks:
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Task']==task)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Task']==task)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['Task']==task)])
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Task']==task)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Task']==task)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['Task']==task)])
	colors= sns.color_palette(['#3F6075', '#FFC61E'])
	sns.set(style="whitegrid")
	# sns.plt.figure(figsize=(51.2,22.8))
	sns.boxplot(data=df[df['Attack Type']!='None'],x='Task',hue='Attack Type',y="Sum of Shortest Paths",palette=colors)
	for i,task in enumerate(tasks):
		stat = tstatfunc(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Task']==task)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Task']==task)])
		maxvaly = np.nanmax(np.array(df['Sum of Shortest Paths'][(df['Attack Type'] != 'None') & (df['Task']==task)])) + (np.nanstd(np.array(df['Sum of Shortest Paths'][(df['Task']==task)&(df['Attack Type'] != 'None')]))/2)
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.tight_layout()
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/attack_human.pdf',dpi=3600)
	sns.plt.close()
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	sns.boxplot(data=btw_df,x='Task',hue='Node Type',y="Betweenness",palette=colors)
	for i,task in enumerate(tasks):
		stat = tstatfunc(btw_df['Betweenness'][(btw_df['Node Type'] == 'PC') & (btw_df['Task']==task)],btw_df['Betweenness'][(btw_df['Node Type'] == 'Degree') & (btw_df['Task']==task)])
		maxvaly = np.nanmax(np.array(btw_df['Betweenness'][btw_df['Task']==task])) + (np.nanstd(np.array(btw_df['Betweenness'][btw_df['Task']==task]))/2)
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/humanbtw.pdf',dpi=3600)
	sns.plt.close()
	scipy.stats.ttest_ind(df['Sum of Shortest Paths'][df['Attack Type'] == 'PC Rich Club'],df['Sum of Shortest Paths'][df['Attack Type'] == 'Degree Rich Club'])

def structural_attacks(networks=['macaque','c_elegans','power_grid','air_traffic']):
	networks=['macaque','c_elegans','air_traffic','power_grid']
	try:
		df=pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/stuc_attack')
		btw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/struc_bwt')
		intdf = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/struc_int')
	except:
		intdf = pd.DataFrame()
		df = pd.DataFrame(columns=['Attack Type','Sum of Shortest Paths','network'])
		btw_df = pd.DataFrame(columns=['Betweenness','Node Type','network'])
		for network in networks:
			if network == 'macaque':
				matrix = loadmat('/home/despoB/mb3152/dynamic_mod/%s.mat'%(network))['CIJ']
				temp_matrix = matrix.copy()
				graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=1.,mst=True)
				vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
			# if network == 'cat':
			# 	matrix = loadmat('%s.mat'%(network))['CIJall']
			# 	temp_matrix = matrix.copy()
			# 	graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=1.,mst=True)
			if network == 'c_elegans':
				graph = Graph.Read_GML('/home/despoB/mb3152/dynamic_mod/celegansneural.gml')
				graph.es["weight"] = np.ones(graph.ecount())
				matrix = np.array(graph.get_adjacency(attribute='weight').data)
				graph = brain_graphs.matrix_to_igraph(matrix,cost=1.,mst=True)
				vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
			if network == 'power_grid':
				graph = power_rich_club(return_graph=True)
				vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
			if network == 'air_traffic':
				graph = airlines_RC(return_graph=True)
				v_to_d = []
				for i in range(3281):
					if len(graph.subcomponent(i)) < 3000:
						v_to_d.append(i)
				graph.delete_vertices(v_to_d)
				vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
			print 'Rich Club Intersection'
			sys.stdout.flush()
			inters = rich_club_intersect(graph.copy(),(graph.vcount())-(graph.vcount()/5),vc)
			temp_df = pd.DataFrame(columns=["Percent Overlap", 'Percent Community, PC','Percent Community, Degree','Task'],index=np.arange(1))
			temp_df["Percent Overlap"] = inters[0]
			temp_df['Percent Community, PC'] = inters[1]
			temp_df['Percent Community, Degree'] = inters[2]
			temp_df['network'] = network
			intdf = intdf.append(temp_df)		
			variables = []
			for i in np.arange(20):
				if network == 'power_grid':
					variables.append([graph.copy(),vc,i])
					continue
				if network == 'air_traffic':
					variables.append([graph.copy(),vc,i])
					continue
				else:
					variables.append([graph.copy(),brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight')),i])
			pool = Pool(20)
			print 'Rich Club Attacks'
			sys.stdout.flush()
			results = pool.map(attack,variables)
			attack_degree_sps = np.array([])
			attack_pc_sps = np.array([])
			healthy_sp = np.array([])
			# attack_degree_mods = np.array([])
			# attack_pc_mods = np.array([])
			# healthy_mods = np.array([])
			for r in results:
				attack_pc_sps = np.append(r[0],attack_pc_sps)
				attack_degree_sps = np.append(r[1],attack_degree_sps)
				healthy_sp = np.append(r[2],healthy_sp)
				# attack_pc_mods = np.append(r[3],attack_pc_mods)
				# attack_degree_mods = np.append(r[4],attack_degree_mods)
				# healthy_mods = np.append(r[5],healthy_mods)
			attack_degree_sps = np.array(attack_degree_sps).reshape(-1)
			attack_pc_sps = np.array(attack_pc_sps).reshape(-1)
			print scipy.stats.ttest_ind(attack_pc_sps,attack_degree_sps)
			# print scipy.stats.ttest_ind(attack_degree_mods,attack_pc_mods)
			b_results = between_community_centrality(graph.copy(),vc)
			print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(b_results[0],b_results[1])	
			sys.stdout.flush()
			sys.stdout.flush()
			hdf = pd.DataFrame()
			task_str = np.ones(len(healthy_sp)).astype(str)
			task_str[:] = network
			hdf['network'] = task_str
			hdf['Sum of Shortest Paths'] = healthy_sp
			hdf['Attack Type'] = 'None'
			df = df.append(hdf)
			d_df = pd.DataFrame()
			task_str = np.ones(len(attack_degree_sps)).astype(str)
			task_str[:] = network
			d_df['network'] = task_str
			d_df['Sum of Shortest Paths'] = attack_degree_sps
			d_df['Attack Type'] = 'Degree Rich Club'
			df = df.append(d_df)
			pc_df = pd.DataFrame()
			task_str = np.ones(len(attack_pc_sps)).astype(str)
			task_str[:] = network
			pc_df['network'] = task_str
			pc_df['Sum of Shortest Paths'] = attack_pc_sps
			pc_df['Attack Type'] = 'PC Rich Club'
			df = df.append(pc_df)
			tbtw_df = pd.DataFrame()
			tbtw_df['Betweenness'] = b_results[0]
			tbtw_df['Node Type'] = 'PC'
			task_str = np.ones(len(b_results[0])).astype(str)
			task_str[:] = network
			tbtw_df['network'] = task_str
			btw_df = btw_df.append(tbtw_df)
			tbtw_df = pd.DataFrame()
			task_str = np.ones(len(b_results[1])).astype(str)
			task_str[:] = network
			tbtw_df['network'] = task_str
			tbtw_df['Betweenness'] = b_results[1]
			tbtw_df['Node Type'] = 'Degree'
			btw_df = btw_df.append(tbtw_df)
		df.to_csv('/home/despoB/mb3152/dynamic_mod/results/stuc_attack')
		btw_df.to_csv('/home/despoB/mb3152/dynamic_mod/results/struc_bwt')
		intdf.to_csv('/home/despoB/mb3152/dynamic_mod/results/struc_int')
	# df['Sum of Shortest Paths'] = df['Sum of Shortest Paths'].astype(float)
	for network in networks:
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network']==network)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network']==network)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['network']==network)])
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network']==network)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network']==network)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['network']==network)])
	sns.set(style="whitegrid",font_scale=1)
	intdf["Percent Overlap"] = intdf["Percent Overlap"].astype(float)
	intdf['Percent Community, PC'] = intdf['Percent Community, PC'].astype(float)
	intdf['Percent Community, Degree'] = intdf['Percent Community, Degree'].astype(float)
	sns.barplot(data=intdf,x='Percent Overlap',y='network')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_overlap_struc.pdf')
	sns.plt.show()
	sns.barplot(data=intdf,x='Percent Community, PC',y='network')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_pc_struc.pdf')
	sns.plt.show()
	sns.barplot(data=intdf,x='Percent Community, Degree',y='network')
	sns.plt.xticks([0,0.2,0.4,0.6,0.8,1])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_degree_struct.pdf')
	sns.plt.show()
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	sns.set(style="whitegrid")
	sns.boxplot(data=df[(df['Attack Type']!='None') & (df.network!='power_grid') & (df.network!='air_traffic') ],palette=colors,x='network',hue='Attack Type',y="Sum of Shortest Paths")
	for i,network in enumerate(networks):
		stat = tstatfunc(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network']==network)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network']==network)])
		maxvaly = np.nanmax(np.array(df['Sum of Shortest Paths'][(df['network']==network) & (df['Attack Type'] != 'None')])) + np.nanstd(np.array(df['Sum of Shortest Paths'][(df['network']==network)&(df['Attack Type'] != 'None')]))/2
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/attack_struc1.pdf',dpi=3600)
	sns.plt.show()
	sns.boxplot(data=df[(df['Attack Type']!='None') & (df.network!='c_elegans') & (df.network!='macaque') & (df.network!='power_grid')],palette=colors,x='network',hue='Attack Type',y="Sum of Shortest Paths")
	for i,network in enumerate(networks):
		stat = tstatfunc(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network']==network)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network']==network)])
		maxvaly = np.nanmax(np.array(df['Sum of Shortest Paths'][(df['network']==network) & (df['Attack Type'] != 'None')])) + np.nanstd(np.array(df['Sum of Shortest Paths'][(df['network']==network)&(df['Attack Type'] != 'None')]))/2
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/attack_struc2.pdf',dpi=3600)
	sns.plt.show()	
	sns.boxplot(data=df[(df['Attack Type']!='None') & (df.network!='c_elegans') & (df.network!='macaque') & (df.network!='air_traffic')],palette=colors,x='network',hue='Attack Type',y="Sum of Shortest Paths")
	for i,network in enumerate(networks):
		stat = tstatfunc(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network']==network)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network']==network)])
		maxvaly = np.nanmax(np.array(df['Sum of Shortest Paths'][(df['network']==network) & (df['Attack Type'] != 'None')])) + np.nanstd(np.array(df['Sum of Shortest Paths'][(df['network']==network)&(df['Attack Type'] != 'None')]))/2
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/attack_struc3.pdf',dpi=3600)
	sns.plt.show()	
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	for network in networks:
		btw_df['Betweenness'][btw_df['network']==network] = (btw_df['Betweenness'][btw_df['network']==network] - np.nanmean(btw_df['Betweenness'][btw_df['network']==network])) / np.nanstd(btw_df['Betweenness'][btw_df['network']==network])
	sns.boxplot(data=btw_df,palette=colors,x='network',hue='Node Type',y="Betweenness")
	sns.plt.ylim(-3,3)
	for i,network in enumerate(networks):
		stat = tstatfunc(btw_df['Betweenness'][(btw_df['Node Type'] == 'PC') & (btw_df['network']==network)],btw_df['Betweenness'][(btw_df['Node Type'] == 'Degree') & (btw_df['network']==network)])
		maxvaly = np.nanmax(np.array(btw_df['Betweenness'][btw_df['network']==network])) + (np.nanstd(np.array(btw_df['Betweenness'][btw_df['network']==network]))/2)
		if maxvaly > 3:
			maxvaly = 3
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/strucbtw.pdf',dpi=3600)
	sns.plt.show()
	for network in networks:
		print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(btw_df['Betweenness'][(btw_df['Node Type'] == 'PC') & (btw_df['network'] == network)],btw_df['Betweenness'][(btw_df['Node Type'] == 'Degree') & (btw_df['network'] == network)])	
	for network in networks:
		print 'Damage, PC Versus Degree', scipy.stats.ttest_ind(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network'] == network)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network'] == network)])	
	print scipy.stats.ttest_ind(df['Sum of Shortest Paths'][df['Attack Type'] == 'PC Rich Club'],df['Sum of Shortest Paths'][df['Attack Type'] == 'Degree Rich Club'])

def fce_attacks():
	worms = ['Worm1','Worm2','Worm3','Worm4']
	try:
		df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/fce_attack')
		btw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/fce_bwt')
	except:
		df = pd.DataFrame(columns=['Attack Type','Sum of Shortest Paths'])
		btw_df = pd.DataFrame(columns=['Betweenness','Node Type','Worm'])
		for worm in worms:
			matrix = np.array(pd.read_excel('pnas.1507110112.sd01.xls',sheetname=worm).corr())[4:,4:]
			variables = []
			temp_matrix = matrix.copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix.copy(),cost=.2,mst=True)
			for i in np.arange(20):
				vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
				variables.append([i,graph.copy(),vc])
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
			variables = []
			for i in np.arange(20):
				# cost = i * 0.01
				cost = .2
				temp_matrix = matrix.copy()
				graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=cost,mst=True)
				variables.append(graph)	
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
			task_str[:] = worm
			hdf['Worm'] = task_str
			hdf['Sum of Shortest Paths'] = healthy_sp
			hdf['Attack Type'] = 'None'
			df = df.append(hdf)
			d_df = pd.DataFrame()
			task_str = np.ones(len(attack_degree_sps)).astype(str)
			task_str[:] = worm
			d_df['Worm'] = task_str
			d_df['Sum of Shortest Paths'] = attack_degree_sps
			d_df['Attack Type'] = 'Degree Rich Club'
			df = df.append(d_df)
			pc_df = pd.DataFrame()
			task_str = np.ones(len(attack_pc_sps)).astype(str)
			task_str[:] = worm
			pc_df['Worm'] = task_str
			pc_df['Sum of Shortest Paths'] = attack_pc_sps
			pc_df['Attack Type'] = 'PC Rich Club'
			df = df.append(pc_df)
			tbtw_df = pd.DataFrame()
			tbtw_df['Betweenness'] = pc_btw
			tbtw_df['Node Type'] = 'PC'
			task_str = np.ones(len(pc_btw)).astype(str)
			task_str[:] = worm
			tbtw_df['Worm'] = task_str
			btw_df = btw_df.append(tbtw_df)
			tbtw_df = pd.DataFrame()
			task_str = np.ones(len(deg_btw)).astype(str)
			task_str[:] = worm
			tbtw_df['Worm'] = task_str
			tbtw_df['Betweenness'] = deg_btw
			tbtw_df['Node Type'] = 'Degree'
			btw_df = btw_df.append(tbtw_df)
		df.to_csv('/home/despoB/mb3152/dynamic_mod/results/fce_attack')
		btw_df.to_csv('/home/despoB/mb3152/dynamic_mod/results/fce_bwt')
	for Worm in worms:
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Worm']==Worm)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Worm']==Worm)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['Worm']==Worm)])
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Worm']==Worm)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Worm']==Worm)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['Worm']==Worm)])
	colors= sns.color_palette(['#3F6075', '#FFC61E'])
	sns.set(style="whitegrid")
	sns.boxplot(data=df[df['Attack Type']!='None'],x='Worm',hue='Attack Type',y="Sum of Shortest Paths",palette=colors)
	for i,Worm in enumerate(worms):
		stat = tstatfunc(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Worm']==Worm)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Worm']==Worm)])
		maxvaly = np.nanmax(np.array(df['Sum of Shortest Paths'][(df['Worm']==Worm) & (df['Attack Type'] != 'None')])) + np.nanstd(np.array(df['Sum of Shortest Paths'][(df['Worm']==Worm)&(df['Attack Type'] != 'None')]))/2
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/attack_fce.pdf',dpi=3600)
	sns.plt.close()
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	sns.boxplot(data=btw_df,x='Worm',hue='Node Type',y="Betweenness",palette=colors)
	for i,Worm in enumerate(worms):
		stat = tstatfunc(btw_df['Betweenness'][(btw_df['Node Type'] == 'PC') & (btw_df['Worm']==Worm)],btw_df['Betweenness'][(btw_df['Node Type'] == 'Degree') & (btw_df['Worm']==Worm)])
		maxvaly = np.nanmax(np.array(btw_df['Betweenness'][btw_df['Worm']==Worm])) + (np.nanstd(np.array(btw_df['Betweenness'][btw_df['Worm']==Worm]))/2)
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/fcebtw.pdf',dpi=3600)
	sns.plt.show()
	for worm in worms:
		print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(btw_df['Betweenness'][(btw_df['Node Type'] == 'PC') & (btw_df['Worm'] == worm)],btw_df['Betweenness'][(btw_df['Node Type'] == 'Degree') & (btw_df['Worm'] == worm)])	
	for worm in worms:
		print 'Damage, PC Versus Degree', scipy.stats.ttest_ind(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Worm'] == worm)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Worm'] == worm)])	
	print scipy.stats.ttest_ind(df['Sum of Shortest Paths'][df['Attack Type'] == 'PC Rich Club'],df['Sum of Shortest Paths'][df['Attack Type'] == 'Degree Rich Club'])

def rich_club_intersect(graph,rank,vc=None):
	if vc == None:
		vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	pc = vc.pc
	mem = np.array(vc.community.membership)
	deg = vc.community.graph.strength(weights='weight')
	deg = np.argsort(deg)[rank:]
	pc = np.argsort(pc)[rank:]
	try:
		overlap = float(len(np.intersect1d(pc,deg)))/len(deg)
	except:
		overlap = 0
	return [overlap,len(np.unique(mem[pc]))/float(len(np.unique(mem))),len(np.unique(mem[deg]))/float(len(np.unique(mem)))]

def human_rich_club():
	"""
	rich club stuff
	"""
	df = pd.DataFrame(columns=["Percent Overlap", 'Percent Community, PC','Percent Community, Degree','Task'])
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
	known_membership,network_names,num_nodes,name_int_dict = network_labels('power')
	network_df = pd.DataFrame(columns=['Task',"Club", "Network",'Number of Club Nodes'])
	for task in tasks:
		atlas = 'power'
		print task
		# subjects = np.array(hcp_subjects).copy()
		# subjects = list(subjects)
		# subjects = remove_missing_subjects(subjects,task,atlas)
		subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',task,atlas))
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		matrices = static_results['matrices']
		avg_pc_normalized_phis = []
		avg_degree_normalized_phis = []
		intersects = []
		average_pc = []
		average_deg = []
		for cost in np.arange(5,21)*0.01:
			temp_matrix = np.nanmean(static_results['matrices'],axis=0).copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix.copy(),cost=cost,mst=True)
			vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
			pc = vc.pc
			pc[np.isnan(pc)] = 0.0
			deg_rank = np.argsort(graph.strength(weights='weight'))[211:]
			pc_rank = np.argsort(pc)[211:]
			# for dcn,rcn in zip(pc_rank,deg_rank):
			# 	network_df = network_df.append({'Task':task,'Club':'Diverse Club','Network':network_names[dcn]},ignore_index=True)
			# 	network_df = network_df.append({'Task':task,'Club':'Rich Club', 'Network':network_names[rcn]},ignore_index=True)
			for network in network_names:
				pc_networks = network_names[pc_rank]==network
				deg_networks = network_names[deg_rank]==network
				network_df = network_df.append({'Task':task,'Club':'Diverse Club','Network':network,'Number of Club Nodes':len(pc_networks[pc_networks==True])},ignore_index=True)
				network_df = network_df.append({'Task':task,'Club':'Rich Club', 'Network':network,'Number of Club Nodes':len(deg_networks[deg_networks==True])},ignore_index=True)

			# community properites of the club
			inters = rich_club_intersect(graph,211)
			temp_df = pd.DataFrame(columns=["Percent Overlap", 'Percent Community, PC','Percent Community, Degree','Task'],index=np.arange(1))
			
			temp_df["Percent Overlap"] = inters[0]
			temp_df['Percent Community, PC'] = inters[1]
			temp_df['Percent Community, Degree'] = inters[2]
			temp_df['Task'] = task
			df = df.append(temp_df)

			## rich club analyses
			pc_emperical_phis = RC(graph, scores=pc).phis()
			# pc_average_randomized_phis = []
			# for i in range(50):
			# 	temp_graph = graph.copy()
			# 	temp_graph = preserve_strength(temp_graph,randomize_topology=True)
			# 	temp_vc = brain_graphs.brain_graph(temp_graph.community_infomap(edge_weights='weight'))
			# 	temp_pc = temp_vc.pc
			# 	temp_pc[np.isnan(temp_pc)] = 0.0
			# 	pc_average_randomized_phis.append(RC(temp_graph,scores=temp_pc).phis())
			# pc_normalized_phis = pc_emperical_phis/np.nanmean(pc_average_randomized_phis,axis=0)
			pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=pc).phis() for i in range(100)],axis=0)
			pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
			
			degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
			average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=graph.strength(weights='weight')).phis() for i in range(100)],axis=0)
			degree_normalized_phis = degree_emperical_phis/average_randomized_phis
			avg_pc_normalized_phis.append(pc_normalized_phis)
			avg_degree_normalized_phis.append(degree_normalized_phis)
		sns.set_style("white")
		sns.set_style("ticks")
		with sns.plotting_context("paper",font_scale=2):	
			# sns.tsplot(np.array(avg_degree_normalized_phis)[:,:-2],color='b',condition='Degree',ci=95)
			# sns.tsplot(np.array(avg_pc_normalized_phis)[:,:-2],color='r',condition='PC',ci=95)
			sns.tsplot(np.array(avg_degree_normalized_phis)[:,:-13],color='b',condition='Degree',ci=95)
			sns.tsplot(np.array(avg_pc_normalized_phis)[:,:-13],color='r',condition='PC',ci=95)
			plt.ylabel('Normalized Rich Club Coefficient')
			plt.xlabel('Rank')
			sns.despine()
			plt.legend()
			plt.tight_layout()
			sns.plt.show()
		# 	if cost == 0.05:
		# 		graph.vs['pc'] = pc
		# 		graph.vs['Community'] = vc.community.membership
		# 		pc_rc = (np.array(pc)>np.percentile(pc,80)).astype(int) * 2
		# 		graph.vs['pc_rc'] = pc_rc
		# 		degree_rc = (np.array(graph.strength(weights='weight')) > np.percentile(graph.strength(weights='weight'),80)).astype(int)
		# 		graph.vs['degree_rc'] = degree_rc
		# 		graph.vs['all_rc'] = degree_rc + pc_rc
		# 		graph.write_gml('human_gephi_%s.gml'%(task))
		# 		1/0
		# 	average_pc.append(vc.pc)
		# 	average_deg.append(graph.strength(weights='weight'))
		# pc = np.nanmean(average_pc,axis=0)
		# deg = np.nanmean(average_deg,axis=0)
		# pc_rc = np.where(pc>=np.percentile(pc,80))[0]
		# deg_rc = np.where(deg>=np.percentile(deg,80))[0]
		# deg_rc_array = np.zeros(len(pc))
		# pc_rc_array = np.zeros(len(pc))
		# deg_rc_array[deg_rc] = 3
		# pc_rc_array[pc_rc] = 6
		# combined_array = deg_rc_array + pc_rc_array

		# import matlab
		# def make_surf_image(values,configs,outfile,drop_zero=False):
		# 	write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		# 	write_df[3] = values
		# 	if drop_zero:
		# 		write_df = write_df[write_df[3]>0]
		# 	write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/rich_club/temp.node',sep='\t',index=False,names=False,header=False)
		# 	node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/rich_club/temp.node'
		# 	surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
		# 	eng.BrainNet_MapCfg(node_file,surf_file,outfile,configs)

		# eng = matlab.engine.start_matlab()
		# eng.addpath('/home/despoB/mb3152/BrainNet/')
		# make_surf_image(deg,configs='/home/despoB/mb3152/BrainNet/pc_values.mat',outfile='/home/despoB/mb3152/dynamic_mod/brain_figures/rich_club/%s_degree.png' %(task))
		# make_surf_image(pc,configs='/home/despoB/mb3152/BrainNet/pc_values.mat',outfile='/home/despoB/mb3152/dynamic_mod/brain_figures/rich_club/%s_pc.png' %(task))
		# make_surf_image(combined_array,configs='/home/despoB/mb3152/BrainNet/pc_values.mat',outfile='/home/despoB/mb3152/dynamic_mod/brain_figures/rich_club/%s_combined_rc.png' %(task),drop_zero=True)


	sns.set_style("white")
	sns.set_style("ticks")
	sns.barplot(data=network_df,x='Network',y='Number of Club Nodes',hue='Club',palette=sns.color_palette(['#7EE062','#050081']))
	sns.plt.xticks(rotation=90)
	sns.plt.ylabel('Mean Number of Club Nodes in Network')
	sns.plt.title('Rich and Diverse Club Network Membership Across Tasks and Costs')
	sns.plt.tight_layout()
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/both_club_membership.pdf',dpi=3600)
	sns.plt.show()
	sns.plt.close()

	sns.set_style("white")
	sns.set_style("ticks")
	with sns.plotting_context("paper",font_scale=2):	
		# sns.tsplot(np.array(avg_degree_normalized_phis)[:,:-2],color='b',condition='Degree',ci=95)
		# sns.tsplot(np.array(avg_pc_normalized_phis)[:,:-2],color='r',condition='PC',ci=95)
		sns.tsplot(np.array(avg_degree_normalized_phis)[:,:-13],color='b',condition='Degree',ci=95)
		sns.tsplot(np.array(avg_pc_normalized_phis)[:,:-13],color='r',condition='PC',ci=95)
		plt.ylabel('Normalized Rich Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend()
		plt.tight_layout()
		sns.plt.show()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_%s_pc_control.pdf'%(task),dpi=3600)
		plt.show()
		plt.close()


	df["Percent Overlap"] = df["Percent Overlap"].astype(float)
	df['Percent Community, PC'] = df['Percent Community, PC'].astype(float)
	df['Percent Community, Degree'] = df['Percent Community, Degree'].astype(float)
	sns.barplot(data=df,x='Percent Overlap',y='Task')
	sns.barplot(data=df,x='Percent Overlap',y='Task',hue='Club',palette=['r','b'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_overlap_human.pdf')
	sns.plt.show()
	sns.barplot(data=df,x='Percent Community, PC',y='Task')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_pc.pdf')
	sns.plt.show()
	sns.barplot(data=df,x='Percent Community, Degree',y='Task')
	sns.plt.xticks([0,0.2,0.4,0.6,0.8,1])
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
	t = v[1]
	task_perf = v[2] 
	train = np.ones(len(pvals)).astype(bool)
	train[t] = False
	clf = linear_model.LinearRegression(fit_intercept=True)
	clf.fit(pvals[train],task_perf[train])
	return clf.predict(pvals[t])

def sm_predict(v):
	pvals = v[0]
	t = v[1]
	task_perf = v[2] 
	train = np.ones(len(pvals)).astype(bool)
	train[t] = False
	pvals = sm.add_constant(pvals)
	r_perf = sm.GLM(task_perf[train],pvals[train]).fit()
	return r_perf.predict(pvals[t])

def corrfunc(x, y, **kws):
	r, _ = pearsonr(x, y)
	ax = plt.gca()
	ax.annotate("r={:.3f}".format(r) + ",p={:.3f}".format(_),xy=(.1, .9), xycoords=ax.transAxes)

def tstatfunc(x, y):
	t, p = scipy.stats.ttest_ind(x,y)
	return "t=%s,p=%s" %(np.around(t,3),np.around(p,5))

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

def test_subs(task,atlas='power'):
	tsubjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',task,atlas))
	tnsubjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_scrub_.2.npy' %('hcp',task,atlas))
	print 'test_subs', (tnsubjects == tsubjects).all()

def supplemental():
	# print 'performance original'
	# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=False).to_csv('/home/despoB/mb3152/dynamic_mod/results/performance_orig.csv')
	# print 'performance scrubbed'
	# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='scrub_.2',control_com=False,control_motion=False).to_csv('/home/despoB/mb3152/dynamic_mod/results/performance_scrubbed.csv')
	print 'performance motion control'
	performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=True).to_csv('/home/despoB/mb3152/dynamic_mod/results/performance_motion_controlled.csv')
	# print 'performance community control'
	# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=True,control_motion=False).to_csv('/home/despoB/mb3152/dynamic_mod/results/performance_community_controlled.csv')
	# print 'correlations original'
	# motion_across_tasks(atlas='power',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='fz',control_com=False,control_motion=False).to_csv('/home/despoB/mb3152/dynamic_mod/results/correlations_original.csv')
	# print 'correlations scrubbed'
	# motion_across_tasks(atlas='power',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='scrub_.2',control_com=False,control_motion=False).to_csv('/home/despoB/mb3152/dynamic_mod/results/correlations_scrubbed.csv')	
	print 'correlations motion control'
	connectivity_across_tasks(atlas='power',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='fz',control_com=False,control_motion=True).to_csv('/home/despoB/mb3152/dynamic_mod/results/correlations_motion_controlled.csv')
	# print 'correlations community control'
	# motion_across_tasks(atlas='power',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='fz',control_com=True,control_motion=False).to_csv('/home/despoB/mb3152/dynamic_mod/results/correlations_community_controlled.csv')

def print_supplemental():
	print 'performance original'
	performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=False)
	print 'performance scrubbed'
	performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='scrub_.2',control_com=False,control_motion=False)
	print 'performance motion control'
	performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=True)
	print 'performance community control'
	performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=True,control_motion=False)
	print 'correlations original'
	motion_across_tasks(atlas='power',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='fz',control_com=False,control_motion=False)
	print 'correlations scrubbed'
	motion_across_tasks(atlas='power',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='scrub_.2',control_com=False,control_motion=False)	
	print 'correlations motion control'
	motion_across_tasks(atlas='power',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='fz',control_com=False,control_motion=True)
	print 'correlations community control'
	motion_across_tasks(atlas='power',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='fz',control_com=True,control_motion=False)

def performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz_wc',control_com=False,control_motion=False):
	import warnings
	warnings.filterwarnings('ignore')
	# tasks=['WM','RELATIONAL','SOCIAL','LANGUAGE']
	# project='hcp'
	# atlas='power'
	known_membership,network_names,num_nodes,name_int_dict = network_labels('power')
	df = pd.DataFrame(columns=['PC','WCD','Task','PCxPerformance','PCxModularity','WCDxPerformance','WCDxModularity'])
	diff_df = pd.DataFrame(columns=['Task','Modularity_Type','Performance'])
	task_perf_df_cols = ['Task','Modularity Increasing Diversity Value','Performance']
	task_perf_df = pd.DataFrame(columns=task_perf_df_cols)
	loo_columns= ['Task','Nodal Predicted Performance','Q Predicted Performance','Mean Nodal Predicted Performance','Mean PC Predicted Performance','Mean WCD Predicted Performance','Performance','Mod Diff','PC Q+ Diff','WCD Q+ Diff','PC Q+/Q- Diff','WCD Q+/Q- Diff','Rest Versus Task']
	loo_df = pd.DataFrame(columns = loo_columns)
	results_df = pd.DataFrame(columns=['Task','Analysis','Statistic', 'Result'])
	for task in tasks:
		print task
		"""
		see which graph metrics correlate with modularity and performance
		"""
		rest_subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/hcp_%s_%s_subs_%s.npy'%('REST',atlas,run_version))
		rest_results = graph_metrics(rest_subjects,'REST',atlas,run_version=run_version)
		rest_subject_pcs = rest_results['subject_pcs'].copy()
		rest_matrices = rest_results['matrices']
		rest_subject_mods = rest_results['subject_mods']
		rest_subject_wmds = rest_results['subject_wmds']
		rest_subjects = rest_results['subjects']
		subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/hcp_%s_%s_subs_%s.npy'%(task,atlas,run_version))
		static_results = graph_metrics(subjects,task,atlas,run_version=run_version)
		subject_pcs = static_results['subject_pcs'].copy()
		subject_wmds = static_results['subject_wmds']
		matrices = static_results['matrices']
		subject_mods = static_results['subject_mods']
		subject_communities = static_results['subject_communities']
		subjects = static_results['subjects']
		if control_motion == True:
			subject_motion = []
			for subject in subjects:
				subject_motion.append(get_sub_motion(subject,task))
			assert np.min(subject_motion) > 0.0
		rest_idx = []
		for i,s in enumerate(rest_subjects):
			if s in subjects:
				rest_idx.append(i)
		# task_idx = []
		# for i,s in enumerate(subjects):
		# 	if s in rest_subjects:
		# 		task_idx.append(i)
		# assert (rest_subjects[rest_idx] == subjects[task_idx]).all()
		# going to need somtehing fancy, since you want all task subjects for some analyses, but some missing rest because of scrubbing, so need to do something for those analyses.
		# might need a special delete below for doing rest to tasks analyses for scrubbing.
		if run_version != 'scrub_.2':
			for i,s in enumerate(subjects):
				assert s in rest_subjects
			assert (rest_subjects[rest_idx] == subjects).all()
		task_perf = task_performance(subjects,task)
		if task == 'SOCIAL':
			task_perf[task_perf==1] = np.nan
		assert subject_pcs.shape[0] == len(subjects)
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
		mod_wmd_corr = np.zeros(subject_wmds.shape[1])
		for i in range(subject_wmds.shape[1]):
			mod_wmd_corr[i] = nan_pearsonr(subject_mods,subject_wmds[:,i])[0]
		
		"""
		predict performance using high and low PCS values. 
		"""
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
		if control_motion == True:
			subject_motion = np.delete(np.array(subject_motion),to_delete)
		subject_communities = np.delete(subject_communities,to_delete)
		task_perf = scipy.stats.zscore(task_perf)
		subject_pcs[np.isnan(subject_pcs)] = 0.0
		rest_subject_pcs[np.isnan(rest_subject_pcs)] = 0.0
		subject_wmds[np.isnan(subject_wmds)] = 0.0

		# if control_com == True and control_motion == True:
		# 	model_vars = np.array([subject_motion,subject_communities]).transpose()
		# 	subject_mods = sm.GLM(subject_mods,sm.add_constant(model_vars)).fit().resid_response
		# 	assert np.isclose(0.0,pearsonr(subject_mods,subject_motion)[0]) == True
		# 	assert np.isclose(0.0,pearsonr(subject_mods,subject_communities)[0]) == True
		# if control_com == True and control_motion == False:
		# 	subject_mods = sm.GLM(subject_mods,sm.add_constant(subject_communities)).fit().resid_response
		# 	assert np.isclose(0.0,pearsonr(subject_mods,subject_communities)[0]) == True
		# if control_com == False and control_motion == True:
		# 	subject_mods = sm.GLM(subject_mods,sm.add_constant(subject_motion)).fit().resid_response
		# 	assert np.isclose(0.0,pearsonr(subject_mods,subject_motion)[0]) == True

		predict_nodes = np.where(mod_pc_corr>0.0)[0]
		local_predict_nodes = np.where(mod_pc_corr<0.0)[0]
		wmd_predict_nodes = np.where(mod_wmd_corr<0.0)[0]
		wmd_local_predict_nodes = np.where(mod_wmd_corr>0.0)[0]

		if control_com == True and control_motion == True:
			model_vars = np.array([subject_motion,subject_communities]).transpose()
			task_perf = sm.GLM(task_perf,sm.add_constant(model_vars)).fit().resid_response
			assert np.isclose(0.0,pearsonr(task_perf,subject_motion)[0]) == True
			assert np.isclose(0.0,pearsonr(task_perf,subject_communities)[0]) == True
		if control_com == True and control_motion == False:
			task_perf = sm.GLM(task_perf,sm.add_constant(subject_communities)).fit().resid_response
			assert np.isclose(0.0,pearsonr(task_perf,subject_communities)[0]) == True
		if control_com == False and control_motion == True:
			task_perf = sm.GLM(task_perf,sm.add_constant(subject_motion)).fit().resid_response
			assert np.isclose(0.0,pearsonr(task_perf,subject_motion)[0]) == True
		# task_perf = sm.GLM(task_perf,sm.add_constant(subject_mods)).fit().resid_response

		"""
		prediction / cross validation
		"""

		# mod_diff = subject_mods - rest_subject_mods
		# pvals = np.array(subject_mods - rest_subject_mods)
		# pvals = pvals.reshape(pvals.shape[0],1)
		# vs = []
		# for t in range(pvals.shape[0]):
		# 	train = np.ones(len(pvals)).astype(bool)
		# 	train[t] = False
		# 	vs.append([pvals,train,t,task_perf])
		# pool = Pool(20)
		# q_diff_prediction = pool.map(predict,vs)
		# print 'Q Difference Prediction of Performance, LOO: ', pearsonr(np.array(q_diff_prediction).reshape(-1),task_perf)

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
		pvals = scipy.stats.zscore(pvals,axis=0)
		vs = []
		for t in range(pvals.shape[0]):
			vs.append([pvals.copy(),t,task_perf])
		pool = Pool(5)
		mean_nodal_prediction = pool.map(sm_predict,vs)
		result = pearsonr(np.array(mean_nodal_prediction).reshape(-1),task_perf)
		print 'Mean Nodal Prediction of Performance, LOO: ', result
		results_df = results_df.append({'Analysis':'Mean Nodal Prediction of Performance, LOO,','Task':task,'Statistic':'R, P','Result': '%s, %s'%(np.round(result[0],3),np.round(result[1],10))},ignore_index=True)
		continue

		pvals = np.concatenate([subject_pcs,subject_wmds],axis=1)
		pvals = scipy.stats.zscore(pvals,axis=0)
		vs = []
		for t in range(pvals.shape[0]):
			# train = np.ones(len(pvals)).astype(bool)
			# train[t] = False
			# vs.append([pvals,train,t,task_perf])
			vs.append([pvals.copy(),t,task_perf])
		pool = Pool(20)
		nodal_prediction = pool.map(predict,vs)
		print 'Nodal (PC and WMD scores) Prediction of Performance, LOO: ', pearsonr(np.array(nodal_prediction).reshape(-1),task_perf)
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
		pvals = scipy.stats.zscore(pvals,axis=0)
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
		pvals = scipy.stats.zscore(pvals,axis=0)
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

	return results_df
	# loo_df.to_csv('/home/despoB/mb3152/dynamic_mod/loo_df.csv')
	# df.to_csv('/home/despoB/mb3152/dynamic_mod/df.csv')

	"""
	PRETTY FIGURES 
	"""
	# loo_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/loo_df.csv')
	# df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/df.csv')


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

def c_elegans_rich_club(plt_mat=False):
	worms = ['Worm1','Worm2','Worm3','Worm4']
	df = pd.DataFrame(columns=["Percent Overlap", 'Percent Community, PC','Percent Community, Degree','Worm'])
	plt_mat = False
	for worm in worms:
		# matrix = np.arctanh(np.array(pd.read_excel('pnas.1507110112.sd01.xls',sheetname=worm).corr())[4:,4:])
		matrix = np.array(pd.read_excel('pnas.1507110112.sd01.xls',sheetname=worm).corr())[4:,4:]
		avg_pc_normalized_phis = []
		avg_degree_normalized_phis = []
		print matrix.shape
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
			inters = rich_club_intersect(graph,int(graph.vcount()*.8))
			temp_df = pd.DataFrame(columns=["Percent Overlap", 'Percent Community, PC','Percent Community, Degree','Worm'],index=np.arange(1))
			temp_df["Percent Overlap"] = inters[0]
			temp_df['Percent Community, PC'] = inters[1]
			temp_df['Percent Community, Degree'] = inters[2]
			temp_df['Worm'] = worm
			df = df.append(temp_df)

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
		with sns.plotting_context("paper",font_scale=1):	
			sns.tsplot(np.array(avg_degree_normalized_phis)[:,:-int(graph.vcount()/20.)],color='b',condition='Rich Club',ci=95)
			sns.tsplot(np.array(avg_pc_normalized_phis)[:,:-int(graph.vcount()/20.)],color='r',condition='Diverse Club',ci=95)
			plt.ylabel('Normalized Club Coefficient')
			plt.xlabel('Rank')
			sns.despine()
			plt.legend()
			plt.tight_layout()
			plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_%s.pdf'%(worm),dpi=3600)
			plt.close()
	sns.set_style("white")
	sns.set_style("ticks")
	df["Percent Overlap"] = df["Percent Overlap"].astype(float)
	df['Percent Community, PC'] = df['Percent Community, PC'].astype(float)
	df['Percent Community, Degree'] = df['Percent Community, Degree'].astype(float)
	sns.barplot(data=df,x='Percent Overlap',y='Worm')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_overlap_fce.pdf')
	sns.plt.show()
	sns.barplot(data=df,x='Percent Community, PC',y='Worm')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_pc_fce.pdf')
	sns.plt.show()
	sns.barplot(data=df,x='Percent Community, Degree',y='Worm')
	sns.plt.xticks([0,0.2,0.4,0.6,0.8,1])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_degree_fce.pdf')	
	sns.plt.show()
	
	# degree_normalized_phis = np.nanmean(avg_degree_normalized_phis,axis=0)
	# pc_normalized_phis = np.nanmean(avg_pc_normalized_phis,axis=0)

def plot_reordered_matrix(matrix,membership=None):
	import matlab
	eng = matlab.engine.start_matlab()
	eng.addpath('/home/despoB/mb3152/brain_graphs/BCT/2016_01_16_BCT/')
	r_matrix = eng.reorderMAT(matlab.double(matrix.tolist()),10,'line')
	if type(membership) != 'NoneType':
		membership = np.array(membership) + 1
		m_matrix = eng.reorder_mod(r_matrix,matlab.double(membership.tolist()))

def power_rich_club(return_graph=False):
	graph = Graph.Read_GML('/home/despoB/mb3152/dynamic_mod/power.gml')
	graph.es["weight"] = np.ones(graph.ecount())
	if return_graph == True:
		return graph
	# inters = rich_club_intersect(graph,(graph.vcount())-graph.vcount()/5)
	# variables = []
	# for i in np.arange(20):
	# 	temp_matrix = matrix.copy()
	# 	graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=1.,mst=True)
	# 	variables.append(graph)		
	# pool = Pool(20)
	# results = pool.map(attack,variables)
	# attack_degree_sps = np.array([])
	# attack_pc_sps = np.array([])
	# healthy_sp = np.array([])
	# for r in results:
	# 	attack_pc_sps = np.append(r[0],attack_pc_sps)
	# 	attack_degree_sps = np.append(r[1],attack_degree_sps)
	# 	healthy_sp = np.append(r[2],healthy_sp)
	# attack_degree_sps = np.array(attack_degree_sps).reshape(-1)
	# attack_pc_sps = np.array(attack_pc_sps).reshape(-1)
	# print scipy.stats.ttest_ind(attack_pc_sps,attack_degree_sps)
	# btw = between_community_centrality(graph)
	# scipy.stats.ttest_ind(btw[0],btw[1])

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
		sns.tsplot(np.array(degree_normalized_phis)[:-int(graph.vcount()/20.)],color='b',condition='Rich Club',ci=99)
		sns.tsplot(np.array(pc_normalized_phis)[:-int(graph.vcount()/20.)],color='r',condition='Diverse Club',ci=99)
		plt.ylabel('Normalized Club Coefficient')
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
	print graph.vcount(), graph.ecount()
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
		sns.tsplot(np.array(degree_normalized_phis)[:-int(graph.vcount()/20.)],color='b',condition='Rich Club',ci=99)
		sns.tsplot(np.array(pc_normalized_phis)[:-int(graph.vcount()/20.)],color='r',condition='Diverse Club',ci=99)
		plt.ylabel('Normalized Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_structural.pdf',dpi=3600)
		plt.show()

def macaque_rich_club():
	matrix = loadmat('%s.mat'%('macaque'))['CIJ']
	graph = brain_graphs.matrix_to_igraph(matrix,cost=1.)
	print graph.vcount(), graph.ecount()
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
		sns.tsplot(degree_normalized_phis[:-int(graph.vcount()/20.)],color='b',condition='Rich Club',ci=90)
		sns.tsplot(pc_normalized_phis[:-int(graph.vcount()/20.)],color='r',condition='Diverse Club',ci=90)
		plt.ylabel('Normalized Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_macaque.pdf',dpi=3600)
		plt.show()
		# plt.close()

def airlines_RC(return_graph=False):
	vs = []
	sources = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[3].values
	dests = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[5].values
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
	sources = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[3].values
	dests = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[5].values
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
	if return_graph == True:
		return graph
	# inters = rich_club_intersect(graph,(graph.vcount())-graph.vcount()/5)
	# variables = []
	# for i in np.arange(20):
	# 	temp_matrix = matrix.copy()
	# 	graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=1.,mst=True)
	# 	variables.append(graph)		
	# pool = Pool(20)
	# results = pool.map(attack,variables)
	# attack_degree_sps = np.array([])
	# attack_pc_sps = np.array([])
	# healthy_sp = np.array([])
	# for r in results:
	# 	attack_pc_sps = np.append(r[0],attack_pc_sps)
	# 	attack_degree_sps = np.append(r[1],attack_degree_sps)
	# 	healthy_sp = np.append(r[2],healthy_sp)
	# attack_degree_sps = np.array(attack_degree_sps).reshape(-1)
	# attack_pc_sps = np.array(attack_pc_sps).reshape(-1)
	# print scipy.stats.ttest_ind(attack_pc_sps,attack_degree_sps)
	# btw = between_community_centrality(graph)
	# print scipy.stats.ttest_ind(btw[0],btw[1])

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
		sns.tsplot(np.array(degree_normalized_phis)[:-int(graph.vcount()/20.)],color='b',condition='Rich Club',ci=99)
		sns.tsplot(np.array(pc_normalized_phis)[:-int(graph.vcount()/20.)],color='r',condition='Diverse Club',ci=99)
		plt.ylabel('Normalized Club Coefficient')
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

# task = 'WM'
# atlas = 'power'
# subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',task,atlas))
# graph_metrics(subjects,task,atlas,project='hcp',run_version='fz_scrubbed',run=True)
# motion_across_tasks()
# performance_across_tasks()
# structural_attacks(networks=['macaque','c_elegans','power_grid','air_traffic'])
# targeted_attacks()
# performance_across_tasks(atlas='power',tasks=[sys.argv[1]])
# performance_across_tasks()
# split_connectivity_across_tasks()
# human_attacks()
# fce_attacks()
"""
SGE Inputs
"""

# atlas = 'power'
# tasks = ['WM','GAMBLING','RELATIONAL','LANGUAGE','SOCIAL','MOTOR','REST']
# for task in tasks:
# 	print task
# 	subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',task,atlas))
# 	for subject in subjects:
# 		s_matrix = []
# 		files = glob.glob('/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_*%s*_matrix_scrubbed_0.2.npy'%(atlas,subject,atlas,task))
# 		# files = glob.glob('/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_*%s*_matrix.npy'%(atlas,subject,atlas,task))
# 		if len(files) < 2:
# 			print subject,task
# 			1/0
# 		for f in files:
# 			f = np.load(f)
# 			np.fill_diagonal(f,0.0)
# 			f[np.isnan(f)] = 0.0
# 			f = np.arctanh(f)
# 			if np.isfinite(f).all() == False:
# 				continue
# 			s_matrix.append(f.copy())
# 		s_matrix = np.nanmean(s_matrix,axis=0)
# 		assert s_matrix.shape == (264,264)
# 		for cost in np.array(range(5,16))*0.01:
# 			temp_matrix = s_matrix.copy()
# 			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True)
# 			assert np.diff([cost,graph.density()])[0] < .005
# print_supplemental()
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
		# subjects = remove_missing_subjects(list(np.array(hcp_subjects).copy()),sys.argv[2],sys.argv[3])
		# subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',sys.argv[2],sys.argv[3]))
		# graph_metrics(subjects,task=sys.argv[2],atlas=sys.argv[3],run_version='fz_wc',run=True)
		subjects = []
		dirs = os.listdir('/home/despoB/connectome-data/')
		for s in dirs:
			try: int(s)
			except: continue
			subjects.append(str(s))
		graph_metrics(subjects,task=sys.argv[2],atlas=sys.argv[3],run_version='HCP_900',run=True)
	if sys.argv[1] == 'make_matrix':
		subject = str(sys.argv[2])
		task = str(sys.argv[3])
		atlas = str(sys.argv[4])
		make_static_matrix(subject,task,'hcp',atlas)
	if sys.argv[1] == 'calc_motion':
		subject = str(sys.argv[2])
		task = str(sys.argv[3])
		run_fd(subject,task)
	if sys.argv[1] == 'check_norm':
		atlas = sys.argv[3]
		task = sys.argv[2]
		# subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',sys.argv[2],sys.argv[3]))
		subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_scrub_.2.npy' %('hcp',sys.argv[2],sys.argv[3]))
		check_scrubbed_normalize(subjects,task,atlas='power')
		print 'done checkin, all good!'
	if sys.argv[1] == 'mediation':
		local_mediation(sys.argv[2]) 


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

