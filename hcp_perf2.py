#!/home/despoB/mb3152/anaconda2/bin/python
import brain_graphs
import pandas as pd
import matlab
import matlab.engine
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
from sklearn.neural_network import MLPRegressor
from itertools import combinations, permutations
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering
import glob
import math
import matplotlib.patches as patches
from collections import Counter
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib import patches
plt.rcParams['pdf.fonttype'] = 42
path = '/home/despoB/mb3152/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/Helvetica.ttf'
prop = mpl.font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = prop.get_name()
import seaborn as sns
import powerlaw
from richclub import preserve_strength, RC
from multiprocessing import Pool
sys.path.append('/home/despoB/mb3152/dynamic_mod/')
from sklearn import linear_model, metrics
import random
global hcp_subjects
hcp_subjects = os.listdir('/home/despoB/connectome-data/')
hcp_subjects.sort()
# global pc_vals 
# global fit_matrices
# global task_perf
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation
from scipy import stats, linalg
global homedir
# homedir = '/Users/Maxwell/HWNI/'
homedir = '/home/despoB/mb3152/'
import multiprocessing

from sklearn.decomposition import PCA,FastICA,FactorAnalysis
from sklearn.cross_decomposition import CCA
import copy

from quantities import millimeter
def mm_2_inches(mm):
	mm = mm * millimeter
	mm.units = 'inches'
	return mm.item()

def alg_compare_multi(matrix):
	alg1mods = []
	alg2mods = []
	alg3mods = []
	for cost in np.array(range(5,16))*0.01:
		temp_matrix = matrix.copy()
		graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True,mst=True)
		assert np.diff([cost,graph.density()])[0] < .01
		alg1mods.append(graph.community_infomap(edge_weights='weight').modularity)
		alg2mods.append(graph.community_multilevel(weights='weight').modularity)
		alg3mods.append(graph.community_fastgreedy(weights='weight').as_clustering().modularity)
	alg1mods = np.nanmean(alg1mods)
	alg2mods = np.nanmean(alg2mods)
	alg3mods = np.nanmean(alg3mods)
	return [alg1mods,alg2mods,alg3mods]

def alg_compare(subjects,homedir=homedir):
	task = 'REST'
	atlas = 'power'
	project='hcp'
	matrices = []
	for subject in subjects:
		s_matrix = []
		files = glob.glob('%sdynamic_mod/%s_matrices/%s_%s_*%s*_matrix.npy'%(homedir,atlas,subject,atlas,task))
		for f in files:
			f = np.load(f)
			np.fill_diagonal(f,0.0)
			f[np.isnan(f)] = 0.0
			f = np.arctanh(f)
			s_matrix.append(f.copy())
		if len(s_matrix) == 0:
			continue
		s_matrix = np.nanmean(s_matrix,axis=0)
		matrices.append(s_matrix.copy())
	pool = Pool(40)
	results = pool.map(alg_compare_multi,matrices)
	np.save('%sdynamic_mod/results/alg_compare.npy'%(homedir),results)

def alg_plot():
	sns.set_style("white")
	sns.set_style("ticks")
	d = np.load('%sdynamic_mod/results/alg_compare.npy'%(homedir))
	df = pd.DataFrame(columns=['Q','Community Algorithm'])
	for i,s in enumerate(d):
		df = df.append({"Q":s[0],'Community Algorithm':'InfoMap','subject':i},ignore_index=True)
		df = df.append({"Q":s[1],'Community Algorithm':'Louvain','subject':i},ignore_index=True)
		df = df.append({"Q":s[2],'Community Algorithm':'Fast Greedy','subject':i},ignore_index=True)
	ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
	ax2 = plt.subplot2grid((3,3), (1, 0))
	ax3 = plt.subplot2grid((3,3), (1, 1))
	ax4 = plt.subplot2grid((3,3), (1, 2))
	sns.set_style("white")
	sns.set_style("ticks")
	sns.set(context="paper",font='Helvetica',font_scale=1.2)
	sns.violinplot(data=df,inner='quartile',y='Q',x='Community Algorithm',palette=sns.color_palette("cubehelix", 8)[-3:],ax=ax1)
	sns.plt.legend(bbox_to_anchor=[1,1.05],columnspacing=10)
	ax1.set_title('Q Values Across Different Algorithms')
	axes = [ax1,ax2,ax3]
	for x,ax in zip(combinations(np.unique(df['Community Algorithm']),2),[ax2,ax3,ax4]):
		print x[0],x[1]
		print pearsonr(df.Q[df['Community Algorithm']==x[0]],df.Q[df['Community Algorithm']==x[1]])
		print scipy.stats.ttest_ind(df.Q[df['Community Algorithm']==x[0]],df.Q[df['Community Algorithm']==x[1]])
		sns.regplot(df.Q[df['Community Algorithm']==x[0]],df.Q[df['Community Algorithm']==x[1]],ax=ax,color=sns.dark_palette("muted purple", input="xkcd")[-1])
		ax.set_xlabel(x[0] + ' Q')
		ax.set_ylabel(x[1] + ' Q')

	sns.plt.show()
	plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/alg_compare.pdf',dpi=3600)
	plt.close()

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

def check_motion(subjects):
	for subject in subjects:
	    e = np.load('/home/despoB/mb3152/dynamic_mod/component_activation/%s_12_False_engagement.npy' %(subject))
	    m = np.loadtxt('/home/despoB/mb3152/data/nki_data/preprocessed/pipeline_comp_cor_and_standard/%s_session_1/frame_wise_displacement/_scan_RfMRI_mx_645_rest/FD.1D'%(subject))
	    print pearsonr(m,np.std(e.reshape(900,12),axis=1))

def plot_corr_matrix(matrix,membership,colors,out_file=None,reorder=True,line=False,rectangle=False,draw_legend=False,colorbar=False):	
	"""
	matrix: square, whatever you like
	membership: the community (or whatever you like of each node in the matrix)
	colors: the colors of each node in the matrix (same order as membership)
	out_file: save the file here, will supress plotting, do None if you want to plot it.
	line: draw those little lines to divide up communities
	rectangle: draw colored rectangles around each community
	draw legend: draw legend...
	colorbar: colorbar...
	"""
	if reorder == True:
		swap_dict = {}
		index = 0
		corr_mat = np.zeros((matrix.shape))
		names = []
		x_ticks = []
		y_ticks = []
		reordered_colors = []
		for i in np.unique(membership):
			for node in np.where(membership==i)[0]:
				swap_dict[node] = index
				index = index + 1
				names.append(membership[node])
				reordered_colors.append(colors[node])
		for i in range(len(swap_dict)):
			for j in range(len(swap_dict)):
				corr_mat[swap_dict[i],swap_dict[j]] = matrix[i,j]
				corr_mat[swap_dict[j],swap_dict[i]] = matrix[j,i]
		colors = reordered_colors
		membership = np.array(names)
	else:
		corr_mat = matrix
	sns.set(style='dark',context="paper",font='Helvetica',font_scale=1.2)
	std = np.nanstd(corr_mat)
	mean = np.nanmean(corr_mat)
	fig = sns.clustermap(corr_mat,yticklabels=[''],xticklabels=[''],cmap=sns.diverging_palette(260,10,sep=10, n=20,as_cmap=True),rasterized=True,col_colors=colors,row_colors=colors,row_cluster=False,col_cluster=False,**{'vmin':mean - (std*2),'vmax':mean + (std*2),'figsize':(15.567,15)})
	ax = fig.fig.axes[4]
	# Use matplotlib directly to emphasize known networks
	if line == True or rectangle == True:
		if len(colors) != len(membership):
			colors = np.arange(len(membership))
		for i,network,color, in zip(np.arange(len(membership)),membership,colors):
			if network != membership[i - 1]:
				if len(colors) != len(membership):
					color = 'white'
				if rectangle == True:
					ax.add_patch(patches.Rectangle((i+len(membership[membership==network]),264-i),len(membership[membership==network]),len(membership[membership==network]),facecolor="none",edgecolor=color,linewidth="2",angle=180))
				if line == True:
					ax.axhline(len(membership) - i, c=color,linewidth=.5,label=network)
					ax.axhline(len(membership) - i, c='black',linewidth=.5)
					ax.axvline(i, c='black',linewidth=.5)
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
		plt.savefig(out_file,dpi=600)
		plt.close()
	if out_file == None:
		plt.show()
	return fig

def plot_corr_matrix2(matrix,membership):	
	"""
	matrix: square, whatever you like
	membership: the community (or whatever you like of each node in the matrix)
	colors: the colors of each node in the matrix (same order as membership)
	out_file: save the file here, will supress plotting, do None if you want to plot it.
	line: draw those little lines to divide up communities
	rectangle: draw colored rectangles around each community
	draw legend: draw legend...
	colorbar: colorbar...
	"""
	sns.set(style='dark',context="paper",font='Helvetica',font_scale=1.2)
	std = np.nanstd(matrix)
	mean = np.nanmean(matrix)
	np.fill_diagonal(matrix,0.0)
	fig = sns.heatmap(matrix,yticklabels=[''],xticklabels=[''],cmap=sns.diverging_palette(260,10,sep=10, n=20,as_cmap=True),rasterized=True,**{'vmin':mean - (std*2),'vmax':mean + (std*2)})
	# Use matplotlib directly to emphasize known networks
	for i,network in zip(np.arange(len(membership)),membership):
		if network != membership[i - 1]:
			fig.figure.axes[0].add_patch(patches.Rectangle((i+len(membership[membership==network]),len(membership)-i),len(membership[membership==network]),len(membership[membership==network]),facecolor="none",edgecolor='black',linewidth="2",angle=180))
	sns.plt.show()

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

def participation_coef(W, ci, degree='undirected'):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    degree : str
        Flag to describe nature of graph 'undirected': For undirected graphs
                                         'in': Uses the in-degree
                                         'out': Uses the out-degree
    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''
    if degree == 'in':
        W = W.T

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P


def check_sym():
	known_membership,network_names,num_nodes,name_int_dict = network_labels('power')
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
	for task in tasks:
		print task
		subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_%s.npy' %('hcp',task,'power','fz'))
		for subject in subjects:
			print task,subject
			s_matrix = []
			files = glob.glob('/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_*%s*_matrix.npy'%('power',subject,'power',task))
			for f in files:
				f = np.load(f)
				np.fill_diagonal(f,0.0)
				f[np.isnan(f)] = 0.0
				f = np.arctanh(f)
				s_matrix.append(f.copy())
			s_matrix = np.nanmean(s_matrix,axis=0)
			assert (np.tril(s_matrix,-1) == np.triu(s_matrix,1).transpose()).all()
			graph = brain_graphs.matrix_to_igraph(s_matrix,0.15,binary=False,check_tri=False,interpolation='midpoint',normalize=True,mst=True)
			graph = brain_graphs.brain_graph(VertexClustering(graph,known_membership))
			graph.pc[np.isnan(graph.pc)] = 0.0
			assert np.max(graph.pc) < 1.0
			assert np.isclose(graph.pc,participation_coef(np.array(graph.matrix),np.array(graph.community.membership))).all() == True
			assert np.nansum(np.abs(graph.pc-participation_coef(np.array(graph.matrix),np.array(graph.community.membership)))) < 1e-10



def check_mst(subjects,task,atlas='power'):
	for task in tasks:
		subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_%s.npy' %('hcp',task,atlas,'fz'))
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
				graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=False,interpolation='midpoint',normalize=False,mst=True)
				# assert np.diff([cost,graph.density()])[0] < .005
				# assert graph.is_connected() == True

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
				1/0
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
	tasks = ['REST','WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL']
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
		assert subject_pcs.shape[0] == len(subjects)
		mean_pc = np.nanmean(subject_pcs,axis=0)
		mean_wmd = np.nanmean(subject_wmds,axis=0)
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_mods,subject_pcs[:,i])[0]
		mod_wmd_corr = np.zeros(subject_wmds.shape[1])
		for i in range(subject_wmds.shape[1]):
			mod_wmd_corr[i] = nan_pearsonr(subject_mods,subject_wmds[:,i])[0]
		if driver == 'PC': m = np.load('%s/dynamic_mod/results/full_med_matrix_new_%s.npy'%(homedir,task))
		else: m = np.load('%s/dynamic_mod/results/full_med_matrix_new_%s_wmds.npy'%(homedir,task))
		mean_conn = np.nanmean(matrices,axis=0)
		e_tresh = np.percentile(mean_conn,85)
		for i in range(264):
			real_t = scipy.stats.ttest_ind(np.abs(m)[i][np.argwhere(mean_conn[i]>=e_tresh)][:,:,np.arange(264)!=i].reshape(-1),np.abs(m)[i][np.argwhere(mean_conn[i]<e_tresh)][:,:,np.arange(264)!=i].reshape(-1))[0]
			# real_t = scipy.stats.ttest_ind(m[i][np.argwhere(mean_conn[i]>=e_tresh)][:,:,np.arange(264)!=i].reshape(-1),m[i][np.argwhere(mean_conn[i]<e_tresh)][:,:,np.arange(264)!=i].reshape(-1))[0]
			if mod_pc_corr[i] > 0.0:
				locality_df = locality_df.append({"Node Type":'Connector Hub','t':real_t,'Task':task.capitalize()},ignore_index=True)
			else:
				locality_df = locality_df.append({"Node Type":'Local Node','t':real_t,'Task':task.capitalize()},ignore_index=True)
		locality_df.dropna(inplace=True)
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
		# 	continue
		# 	if task not in ['WM','RELATIONAL','SOCIAL','LANGUAGE']:
		# 		continue
		# 	ijqcorr = nan_pearsonr(matrices[:,i,j],task_perf)[0]
		# 	perf_edge_corr[i,j] = ijqcorr
		# 	perf_edge_corr[j,i] = ijqcorr
		pc_corr_matrix.append(np.nanmean(pc_edge_corr[predict_nodes,:,:],axis=0))
		q_corr_matrix.append(q_edge_corr)
		# if task in ['WM','RELATIONAL','SOCIAL','LANGUAGE']:
			# print nan_pearsonr(perf_edge_corr.reshape(-1),np.nanmean(pc_edge_corr[predict_nodes,:,:],axis=0).reshape(-1))
			# plot_corr_matrix(perf_edge_corr[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_%s_edge_perf_corr_matrix.pdf'%(homedir,task,run_version),reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)
		# plot_corr_matrix(np.nanmean(m[predict_nodes,:,:],axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_%s_%s_mediation_matrix.pdf'%(homedir,task,driver,run_version),reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)
		# plot_corr_matrix(np.nanmean(pc_edge_corr[predict_nodes],axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_%s_pcedge__corr_matrix.pdf'%(homedir,task,run_version),reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)
		# plot_corr_matrix(q_edge_corr[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_%s_%s_qedgecorr_matrix.pdf'%(homedir,task,driver,run_version),reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)
	plot_corr_matrix(np.nanmean(q_corr_matrix,axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_mean_q_corr_matrix.pdf'%(homedir,run_version),reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)
	plot_corr_matrix(np.nanmean(pc_corr_matrix,axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_mean_pc_corr_matrix.pdf'%(homedir,run_version),reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)
	plot_corr_matrix(np.nanmean(m[predict_nodes,:,:],axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file='%s/dynamic_mod/figures/%s_%s_mean_mediation_matrix_withbar.pdf'%(homedir,driver,run_version),reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)	
	# plot_corr_matrix(np.nanmean(pc_corr_matrix,axis=0)[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file=None,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)
	f = sns.plt.figure(figsize=(18,6))
	sns.set_style("white")
	sns.set_style("ticks")
	sns.set(context="paper",font='Helvetica',font_scale=1.2)
	sns.violinplot(data=locality_df[locality_df['Node Type']=='Connector Hub'],x='Task',y='t',hue='Task',inner='quartile',palette=sns.palettes.color_palette('Paired',7))
	sns.plt.ylabel("T Test Values, mediation values of node's nieghbors \n versus mediation of node's non-neighbors")
	sns.plt.legend(bbox_to_anchor=[1,1.05],ncol=7,columnspacing=10)
	sns.plt.savefig('%s/dynamic_mod/figures/%s_mediation_t_test.pdf'%(homedir,run_version))
	sns.plt.show()

	# plot_corr_matrix(mean_conn[:,swap_indices][swap_indices],network_names[swap_indices].copy(),out_file=None,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)
# q_m = np.nanmean(q_corr_matrix,axis=0)
# np.fill_diagonal(q_m,np.nan)
# pc_m = np.nanmean(pc_corr_matrix,axis=0)
# np.fill_diagonal(pc_m,np.nan)
# nan_pearsonr(q_m.flatten(),pc_m.flatten())

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

def hcp_motion(subject,task):
	motion_files = glob.glob('/home/despoB/connectome-raw/%s/MNINonLinear/Results/*%s*/Movement_RelativeRMS_mean.txt'%(subject,task))
	smo = []
	for m in motion_files:
		smo.append(np.nanmean(np.loadtxt(m)))
	smo = np.nanmean(smo)
	return smo

# my_ver = []
# hcp_ver = []
# for s in subjects:
# 	my_ver.append(get_sub_motion(s,''))
# 	hcp_ver.append(hcp_motion(s,''))


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

def individual_differnce_networks(task,atlas='power',run_version='fz'):
	known_membership,network_names,num_nodes,name_int_dict = network_labels(atlas)
	
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

	subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_%s.npy' %('hcp',task,atlas,run_version))
	static_results = graph_metrics(subjects,task,atlas,run_version=run_version)
	matrices = static_results['matrices']
	diff_matrix = np.zeros((264,264))
	for i,j in combinations(range(264),2):
		r = np.nanmean(np.diagonal(generate_correlation_map(matrices[:,i,:].swapaxes(0,1), matrices[:,j,:].swapaxes(0,1))))
		diff_matrix[i,j] = r
		diff_matrix[j,i] = r
	plot_corr_matrix(matrix=diff_matrix[:,swap_indices][swap_indices],membership=network_names[swap_indices].copy(),out_file=None,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)
	plot_corr_matrix(matrix=np.nanmean(matrices,axis=0)[:,swap_indices][swap_indices],membership=network_names[swap_indices].copy(),out_file=None,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)
	plot_corr_matrix(matrix=diff_matrix-np.nanmean(matrices,axis=0)[:,swap_indices][swap_indices],membership=network_names[swap_indices].copy(),out_file=None,reorder=False,colors=colors[swap_indices],line=True,draw_legend=True,rectangle=False)
	for network in np.unique(network_names):
		print network, np.mean((diff_matrix-np.nanmean(matrices,axis=0))[network_names==network][:,network_names!=network])

	for network in np.unique(network_names):
		print network, scipy.stats.ttest_ind((diff_matrix-np.nanmean(matrices,axis=0))[network_names==network][:,network_names!=network].reshape(-1),(diff_matrix-np.nanmean(matrices,axis=0))[network_names==network][:,network_names==network].reshape(-1))

def make_mean_matrix():
	atlas = 'power'
	for task in ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']:
		print task
		matrix = []
		subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_power_subs_fz.npy' %('hcp',task))
		for subject in subjects:
			files = glob.glob('%sdynamic_mod/%s_matrices/%s_%s_*%s*_matrix.npy'%(homedir,atlas,subject,atlas,task))
			for f in files:
				f = np.load(f)
				assert np.nanmax(f) <= 1.
				np.fill_diagonal(f,0.0)
				mmax = np.nanmax(abs(np.tril(f,-1) - np.triu(f,1).transpose()))
				if mmax != 0.0: print subject
				assert np.isclose(mmax,0)
				f = np.arctanh(f)
				matrix.append(f.copy())
		matrix = np.nanmean(matrix,axis=0)
		print np.min(np.max(matrix,axis=1))
		assert (matrix == np.load('/home/despoB/mb3152/diverse_club/graphs/%s.npy'%(task))).all() == True
		# np.save('/home/despoB/mb3152/diverse_club/graphs/%s.npy'%(task),matrix)

def connectivity_across_tasks(atlas='power',project='hcp',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='fz_wc',control_com=False,control_motion=False):
	import matlab
	import matlab.engine	
	eng = matlab.engine.start_matlab()
	eng.addpath('/home/despoB/mb3152/BrainNet/')	
	atlas='power'
	project='hcp'
	tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
	run_version='fz_wc'
	control_com=False
	control_motion=False
	pc_df = pd.DataFrame(columns=['Task','Mean Participation Coefficient','Diversity Facilitated Modularity Coefficient'])
	wmd_df = pd.DataFrame(columns=['Task','Mean Within-Community-Strength','Locality Facilitated Modularity Coefficient'])
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
		mean_wmd = np.nanmean(static_results['subject_wmds'],axis=0)
		mod_pc_corr = np.zeros(subject_pcs.shape[1])
		mod_wmd_corr = np.zeros(subject_pcs.shape[1])
		if control_com == True and control_motion == True:
			model_vars = np.array([subject_motion,subject_communities]).transpose()
			r_mod = sm.GLM(subject_mods,sm.add_constant(model_vars)).fit()
			assert np.isclose(0.0,pearsonr(r_mod.resid_response,subject_motion)[0]) == True
			assert np.isclose(0.0,pearsonr(r_mod.resid_response,subject_communities)[0]) == True
			c_str = 'Motion and Number of Communities'
			subject_mods = r_mod.resid_response
		if control_com == True and control_motion == False:
			r_mod = sm.GLM(subject_mods,sm.add_constant(subject_communities)).fit()
			assert np.isclose(0.0,pearsonr(r_mod.resid_response,subject_communities)[0]) == True
			c_str = 'Number of Communities'
			subject_mods = r_mod.resid_response
		if control_com == False and control_motion == True:
			r_mod = sm.GLM(subject_mods,sm.add_constant(subject_motion)).fit()
			assert np.isclose(0.0,pearsonr(r_mod.resid_response,subject_motion)[0]) == True
			c_str = 'Motion'
			subject_mods = r_mod.resid_response
		for i in range(subject_pcs.shape[1]):
			mod_pc_corr[i] = nan_pearsonr(subject_pcs[:,i],subject_mods)[0]
		for i in range(subject_pcs.shape[1]):
			mod_wmd_corr[i] = nan_pearsonr(subject_wmds[:,i],subject_mods)[0]
		task_str = np.zeros((len(mean_pc))).astype(str)
		task_str[:] = task
		pc_df = pc_df.append(pd.DataFrame(np.array([task_str,mean_pc,mod_pc_corr]).transpose(),columns=['Task','Mean Participation Coefficient','Diversity Facilitated Modularity Coefficient']),ignore_index=True)
		wmd_df = wmd_df.append(pd.DataFrame(np.array([task_str,mean_wmd,mod_wmd_corr]).transpose(),columns=['Task','Mean Within-Community-Strength','Locality Facilitated Modularity Coefficient']),ignore_index=True)
		continue
		import matlab

		# pc values
		write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		pcs = np.nanmean(subject_pcs,axis=0)
		write_df[3] = pcs
		write_df = write_df[write_df[3]>np.percentile(write_df[3].values,80)]
		write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_%s.node'%(task),sep='\t',index=False,names=False,header=False)
		node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_%s.node'%(task)
		surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
		img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/pc_%s.png' %(task)
		configs = '/home/despoB/mb3152/BrainNet/pc_values_thresh.mat'
		eng.BrainNet_MapCfg(node_file,surf_file,configs,img_file)
		#mod pc values
		write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		write_df[3] = mod_pc_corr
		write_df = write_df[write_df[3]>np.percentile(write_df[3].values,80)]
		write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_mod_%s.node'%(task),sep='\t',index=False,names=False,header=False)			
		node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/power_pc_mod_%s.node'%(task)
		surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
		img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/mod_pc_corr_%s.png' %(task)
		configs = '/home/despoB/mb3152/BrainNet/pc_values_thresh.mat'
		eng.BrainNet_MapCfg(node_file,surf_file,configs,img_file)
		# wcd values
		write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		wmds = np.nanmean(subject_wmds,axis=0)
		write_df[3] = wmds
		write_df = write_df[write_df[3]>np.percentile(write_df[3].values,80)]
		write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_wmds_%s.node'%(task),sep='\t',index=False,names=False,header=False)
		node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/power_wmds_%s.node'%(task)
		surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
		img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/wmds_%s.png' %(task)
		configs = '/home/despoB/mb3152/BrainNet/pc_values_thresh.mat'
		eng.BrainNet_MapCfg(node_file,surf_file,configs,img_file)
		#mod wcd values
		write_df = pd.read_csv('/home/despoB/mb3152/BrainNet/Data/ExampleFiles/Power264/Node_Power264.node',header=None,sep='\t')
		write_df[3] = mod_wmd_corr
		write_df = write_df[write_df[3]>np.percentile(write_df[3].values,80)]
		write_df.to_csv('/home/despoB/mb3152/dynamic_mod/brain_figures/power_wmd_mod_%s.node'%(task),sep='\t',index=False,names=False,header=False)			
		node_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/power_wmd_mod_%s.node'%(task)
		surf_file = '/home/despoB/mb3152/BrainNet/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv'
		img_file = '/home/despoB/mb3152/dynamic_mod/brain_figures/mod_wmd_corr_%s.png' %(task)
		configs = '/home/despoB/mb3152/BrainNet/pc_values_thresh.mat'
		eng.BrainNet_MapCfg(node_file,surf_file,configs,img_file)
	return pc_df,wmd_df

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

def connector_mediation(task):
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

def local_mediation(task):
	"""
	264,264,264 matrix, which edges mediate the relationship between WMD and Q
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
	subject_wmds[np.isnan(subject_pcs)] = 0.0
	m = np.zeros((264,264,264))
	pool = Pool(40)
	for n in range(264):
		print n
		sys.stdout.flush()
		variables = []
		for i,j in combinations(range(264),2):
			variables.append(pd.DataFrame(data={'pc':subject_wmds[:,n],'weight':matrices[:,i,j],'q':subject_mods},index=range(len(subject_pcs))))
		results = pool.map(multi_med,variables)
		for r,i in zip(results,combinations(range(264),2)):
			m[n,i[0],i[1]] = r
			m[n,i[1],i[0]] = r
		np.save('/home/despoB/mb3152/dynamic_mod/results/full_med_matrix_new_%s_wmds.npy'%(task),m)

def local_versus_connector_mediation(task):
	locality_df = pd.DataFrame()
	for tasks in ['REST','WM','GAMBLING','SOCIAL','RELATIONAL','MOTOR','LANGUAGE']:
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
		e_tresh = np.percentile(mean_conn,85)
		local = np.load('%s/dynamic_mod/results/full_med_matrix_new_%s_wmds.npy'%(homedir,task))
		connector = np.load('%s/dynamic_mod/results/full_med_matrix_new_%s.npy'%(homedir,task))
		local = np.abs(local)
		connector = np.abs(connector)
		
		for i in range(264):
			if i in np.arange(264)[np.where(mod_wmd_corr>0.0)]:
				real_t = scipy.stats.ttest_ind(local[i][np.argwhere(mean_conn[i]>e_tresh)][:,:,np.arange(264)!=i].reshape(-1),local[i][np.argwhere(mean_conn[i]<e_tresh)][:,:,np.arange(264)!=i].reshape(-1))[0]
				locality_df = locality_df.append({"Node Type":'Local Hub','t':real_t},ignore_index=True)
			if i in np.arange(264)[np.where(mod_pc_corr>0.0)]:
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
					weight_change_matrix_between[n1,n2] = np.nansum(pc_edge_corr[n1][n2][np.where((known_membership!=known_membership[n2])&(np.arange(264)!=n1))])
					weight_change_matrix_within[n1,n2] = np.nansum(pc_edge_corr[n1][n2][np.where((known_membership==known_membership[n2])&(np.arange(264)!=n1))])
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
	# plot_connectivity_results(df[(df['Q+/Q-']=='Q+') &(df['Hub Measure']=='WCD')],"Strength of r's, i's WCD & j's Q",'Average Edge i-j Weight','/home/despoB/mb3152/dynamic_mod/figures/edge_spec_wmdqplus_%s.pdf'%(edge_thresh_val))
	# plot_connectivity_results(df[(df['Q+/Q-']=='Q-') &(df['Hub Measure']=='WCD')],"Strength of r's, i's WCD & j's Q",'Average Edge i-j Weight','/home/despoB/mb3152/dynamic_mod/figures/edge_spec_wmdqminus_%s.pdf'%(edge_thresh_val))
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

def get_power_partition(atlas):
   return np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)

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
	return 

def corrfunc(x, y, **kws):
	r, _ = pearsonr(x, y)
	ax = plt.gca()
	ax.annotate("r={:.3f}".format(r) + ",p={:.3f}".format(_),xy=(.1, .9), xycoords=ax.transAxes)

def tstatfunc(x, y,bc=False):
	t, p = scipy.stats.ttest_ind(x,y)
	if bc != False:
		bfc = np.around((p * bc),5)
		if bfc <= 0.05:
			return "t=%s,p=%s,bf=%s" %(np.around(t,3),np.around(p,5),bfc)
		else:
			return "t=%s" %(np.around(t,3))
	return "t=%s,p=%s" %(np.around(t,3),np.around(p,5))

def plot_connectivity_results(data,x,y,save_str):
	sns.set_style("white")
	sns.set_style("ticks")
	colors = sns.palettes.color_palette('Paired',7)
	colors = np.array(colors)
	sns.set(context="paper",font='Helvetica',style="white",font_scale=1.5)
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(data,col='Task',hue='Task',sharex=False,sharey=False,palette=colors,col_wrap=4)
		g = g.map(sns.regplot,x,y,scatter_kws={'alpha':.50})
		g.map(corrfunc,x,y)
		sns.despine()
		plt.tight_layout()
		plt.savefig(save_str,dpi=3600)
		plt.close()

def plot_results(data,x,y,save_str):
	sns.set(context="paper",font='Helvetica',style="white",font_scale=1.5)
	colors = np.array(sns.palettes.color_palette('Paired',6))
	with sns.plotting_context("paper",font_scale=1):
		g = sns.FacetGrid(data, col='Task', hue='Task',sharex=False,sharey=False,palette=colors[[0,2,4,5]],col_wrap=2)
		g = g.map(sns.regplot,x,y,scatter_kws={'alpha':.95})
		g.map(corrfunc,x,y)
		sns.despine()
		plt.tight_layout()
		plt.savefig(save_str,dpi=3600)
		plt.close()

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

def generate_correlation_map(x, y):
	"""
	Correlate each n with each m.
	----------
	Parameters

	x : np.array, shape N X T.
	y : np.array, shape M X T.
	Returns: np.array, N X M array in which each element is a correlation coefficient.
	----------
	"""
	mu_x = x.mean(1)
	mu_y = y.mean(1)
	n = x.shape[1]
	if n != y.shape[1]:
	    raise ValueError('x and y must ' +
	                     'have the same number of timepoints.')
	s_x = x.std(1, ddof=n - 1)
	s_y = y.std(1, ddof=n - 1)
	cov = np.dot(x,
	             y.T) - n * np.dot(mu_x[:, np.newaxis],
	                              mu_y[np.newaxis, :])
	return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

def super_edge_predict_new(v):

	subject_pcs = v[0]
	subject_wmds = v[1]
	subject_mods = v[2]
	
	rest_subject_pcs = v[3]
	rest_subject_wmds = v[4]
	rest_subject_mods = v[5]
	task_perf = v[6]
	t = v[7]

	task_matrices = v[8]
	rest_matrices = v[9]
	return_features = v[10]
	use_matrix = v[11]

	fit_mask = np.ones((subject_pcs.shape[0])).astype(bool)
	fit_mask[t] = False
	if use_matrix == True:
		flat_matrices = np.zeros((subject_pcs.shape[0],len(np.tril_indices(264,-1)[0])))
		for s in range(subject_pcs.shape[0]):
			m = task_matrices[s]
			flat_matrices[s] = m[np.tril_indices(264,-1)]
		perf_edge_corr = generate_correlation_map(task_perf[fit_mask].reshape(1,-1),flat_matrices[fit_mask].transpose())[0]

		perf_edge_scores = np.zeros((subject_pcs.shape[0]))
		for s in range(subject_pcs.shape[0]):
			perf_edge_scores[s] = pearsonr(flat_matrices[s],perf_edge_corr)[0]
		
		flat_matrices = np.zeros((subject_pcs.shape[0],len(np.tril_indices(264,-1)[0])))
		for s in range(subject_pcs.shape[0]):
			m = rest_matrices[s]
			flat_matrices[s] = m[np.tril_indices(264,-1)]
		rest_perf_edge_corr = generate_correlation_map(task_perf[fit_mask].reshape(1,-1),flat_matrices[fit_mask].transpose())[0]

		rest_perf_edge_scores = np.zeros((subject_pcs.shape[0]))
		for s in range(subject_pcs.shape[0]):
			rest_perf_edge_scores[s] = pearsonr(flat_matrices[s],rest_perf_edge_corr)[0]

	perf_pc_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		perf_pc_corr[i] = nan_pearsonr(task_perf[fit_mask],subject_pcs[fit_mask,i])[0]
	perf_wmd_corr = np.zeros(subject_wmds.shape[1])
	for i in range(subject_wmds.shape[1]):
		perf_wmd_corr[i] = nan_pearsonr(task_perf[fit_mask],subject_wmds[fit_mask,i])[0]
	mod_pc_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_pc_corr[i] = nan_pearsonr(task_perf[fit_mask],rest_subject_pcs[fit_mask,i])[0]
	mod_wmd_corr = np.zeros(subject_wmds.shape[1])
	for i in range(subject_wmds.shape[1]):
		mod_wmd_corr[i] = nan_pearsonr(task_perf[fit_mask],rest_subject_wmds[fit_mask,i])[0]

	task_pc = np.zeros(subject_pcs.shape[0])
	task_wmd = np.zeros(subject_pcs.shape[0])
	for s in range(subject_pcs.shape[0]):
		task_pc[s] = nan_pearsonr(subject_pcs[s],perf_pc_corr)[0]
		task_wmd[s] = nan_pearsonr(subject_wmds[s],perf_wmd_corr)[0]

	rest_pc = np.zeros(subject_pcs.shape[0])
	rest_wmd = np.zeros(subject_pcs.shape[0])
	for s in range(subject_pcs.shape[0]):
		rest_pc[s] = nan_pearsonr(rest_subject_pcs[s],mod_pc_corr)[0]
		rest_wmd[s] = nan_pearsonr(rest_subject_wmds[s],mod_wmd_corr)[0]

	if use_matrix == True: 
		pvals = np.array([rest_pc,rest_wmd,task_pc,task_wmd,rest_perf_edge_scores,perf_edge_scores,rest_subject_mods,subject_mods]).transpose()
		neurons = (8,8,8,)
		# neurons = (8,12,8,12)
	elif use_matrix == False:
		pvals = np.array([rest_pc,rest_wmd,task_pc,task_wmd,rest_subject_mods,subject_mods]).transpose()
		neurons = (6,6,6,)
		# neurons = (6,9,6,9)

		
	train = np.ones(len(pvals)).astype(bool)
	train[t] = False
	model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=neurons,alpha=1e-5,random_state=t)
	model.fit(pvals[train],task_perf[train])
	result = model.predict(pvals[t].reshape(1, -1))[0]
	if return_features == True: 
		return pvals[t],result
	return result

def super_edge_predict(v):

	subject_pcs = v[0]
	subject_wmds = v[1]
	subject_mods = v[2]
	
	rest_subject_pcs = v[3]
	rest_subject_wmds = v[4]
	rest_subject_mods = v[5]
	task_perf = v[6]
	t = v[7]
	neurons = v[8]

	task_matrices = v[9]
	rest_matrices = v[10]
	return_features = v[11]
	use_matrix = v[12]

	fit_mask = np.ones((subject_pcs.shape[0])).astype(bool)
	fit_mask[t] = False

	flat_matrices = np.zeros((subject_pcs.shape[0],len(np.tril_indices(264,-1)[0])))
	for s in range(subject_pcs.shape[0]):
		m = task_matrices[s]
		flat_matrices[s] = m[np.tril_indices(264,-1)]
	perf_edge_corr = generate_correlation_map(task_perf[fit_mask].reshape(1,-1),flat_matrices[fit_mask].transpose())[0]

	perf_edge_scores = np.zeros((subject_pcs.shape[0]))
	for s in range(subject_pcs.shape[0]):
		perf_edge_scores[s] = pearsonr(flat_matrices[s],perf_edge_corr)[0]
	
	flat_matrices = np.zeros((subject_pcs.shape[0],len(np.tril_indices(264,-1)[0])))
	for s in range(subject_pcs.shape[0]):
		m = rest_matrices[s]
		flat_matrices[s] = m[np.tril_indices(264,-1)]
	mod_edge_corr = generate_correlation_map(rest_subject_mods[fit_mask].reshape(1,-1),flat_matrices[fit_mask].transpose())[0]

	mod_edge_scores = np.zeros((subject_pcs.shape[0]))
	for s in range(subject_pcs.shape[0]):
		mod_edge_scores[s] = pearsonr(flat_matrices[s],mod_edge_corr)[0]

	perf_pc_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		perf_pc_corr[i] = nan_pearsonr(task_perf[fit_mask],subject_pcs[fit_mask,i])[0]
	perf_wmd_corr = np.zeros(subject_wmds.shape[1])
	for i in range(subject_wmds.shape[1]):
		perf_wmd_corr[i] = nan_pearsonr(task_perf[fit_mask],subject_wmds[fit_mask,i])[0]
	mod_pc_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		mod_pc_corr[i] = nan_pearsonr(task_perf[fit_mask],rest_subject_pcs[fit_mask,i])[0]
	mod_wmd_corr = np.zeros(subject_wmds.shape[1])
	for i in range(subject_wmds.shape[1]):
		mod_wmd_corr[i] = nan_pearsonr(task_perf[fit_mask],rest_subject_wmds[fit_mask,i])[0]

	task_pc = np.zeros(subject_pcs.shape[0])
	task_wmd = np.zeros(subject_pcs.shape[0])
	for s in range(subject_pcs.shape[0]):
		task_pc[s] = nan_pearsonr(subject_pcs[s],perf_pc_corr)[0]
		task_wmd[s] = nan_pearsonr(subject_wmds[s],perf_wmd_corr)[0]

	rest_pc = np.zeros(subject_pcs.shape[0])
	rest_wmd = np.zeros(subject_pcs.shape[0])
	for s in range(subject_pcs.shape[0]):
		rest_pc[s] = nan_pearsonr(rest_subject_pcs[s],mod_pc_corr)[0]
		rest_wmd[s] = nan_pearsonr(rest_subject_wmds[s],mod_wmd_corr)[0]
	if use_matrix == True: 
		pvals = np.array([rest_pc,rest_wmd,task_pc,task_wmd,mod_edge_scores,perf_edge_scores]).transpose()
		neurons = (6,9,6,9,)
	elif use_matrix == False: 
		pvals = np.array([rest_pc,rest_wmd,task_pc,task_wmd,]).transpose()
		neurons = (4,6,4,6,)
		
	train = np.ones(len(pvals)).astype(bool)
	train[t] = False
	if return_features == True: 
		return pvals[t]
	model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=neurons,alpha=1e-5,random_state=t)
	model.fit(pvals[train],task_perf[train])
	result = model.predict(pvals[t].reshape(1, -1))[0]

	return result

def task_performance(subjects,task):
	df = pd.read_csv('/%s/dynamic_mod/S900_Release_Subjects_Demographics.csv'%(homedir))
	performance = []
	if task == 'WM': 
		wm_df = pd.DataFrame(np.array([df.Subject.values,df['WM_Task_Acc'].values]).transpose(),columns=['Subject','ACC']).dropna()
		for subject in subjects:
			temp_df = wm_df[wm_df.Subject==subject]
			if len(temp_df) == 0:
				performance.append(np.nan)
				continue
			performance.append(temp_df['ACC'].values[0])
	if task == 'RELATIONAL':
		for subject in subjects:
			try:performance.append(df['Relational_Task_Acc'][df.Subject == subject].values[0])
			except: performance.append(np.nan)
	if task == 'LANGUAGE': 
		for subject in subjects:
			try:performance.append(np.nanmax([df['Language_Task_Story_Avg_Difficulty_Level'][df.Subject == subject].values[0],df['Language_Task_Math_Avg_Difficulty_Level'][df.Subject == subject].values[0]]))
			except: performance.append(np.nan)
	if task == 'SOCIAL':
		social_df = pd.DataFrame(np.array([df.Subject,df['Social_Task_TOM_Perc_TOM'],df['Social_Task_Random_Perc_Random']]).transpose(),columns=['Subject','ACC_TOM','ACC_RANDOM']).dropna()
		for subject in subjects:
			temp_df = social_df[social_df.Subject==subject]
			if len(temp_df) == 0:
				performance.append(np.nan)
				continue
			performance.append(np.nanmean([temp_df['ACC_RANDOM'].values[0],temp_df['ACC_TOM'].values[0]]))
	performance = np.array(performance)
	performance[np.where(np.array(subjects).astype(int) == 142626)[0]] = np.nan
	return performance

def behavior(subjects):
	df = pd.read_csv('/%s/dynamic_mod/S900_Release_Subjects_Demographics.csv'%(homedir))
	task_perf = pd.DataFrame(columns=['WM','RELATIONAL','SOCIAL','LANGUAGE'])
	for task in task_perf.columns.values:
		task_perf[task] = task_performance(df.Subject.values,task)
	task_perf['Subject'] =df.Subject.values
	# task_perf = task_perf.dropna()
	fin = pd.merge(task_perf,df,how='outer',on='Subject')
	to_keep = ['MMSE_Score','PicSeq_AgeAdj','CardSort_AgeAdj','Flanker_AgeAdj','PMAT24_A_CR',\
	'ReadEng_AgeAdj','PicVocab_AgeAdj','ProcSpeed_AgeAdj','DDisc_AUC_40K','DDisc_AUC_200',\
	'SCPT_SEN','SCPT_SPEC','IWRD_TOT','ListSort_AgeAdj',\
	'ER40_CR','ER40ANG','ER40FEAR','ER40HAP','ER40NOE','ER40SAD',\
	'AngAffect_Unadj','AngHostil_Unadj','AngAggr_Unadj','FearAffect_Unadj','FearSomat_Unadj','Sadness_Unadj',\
	'LifeSatisf_Unadj','MeanPurp_Unadj','PosAffect_Unadj','Friendship_Unadj','Loneliness_Unadj',\
	'PercHostil_Unadj','PercReject_Unadj','EmotSupp_Unadj','InstruSupp_Unadj'\
	'PercStress_Unadj','SelfEff_Unadj','Endurance_AgeAdj','GaitSpeed_Comp','Dexterity_AgeAdj','Strength_AgeAdj',\
	'NEOFAC_A','NEOFAC_O','NEOFAC_C','NEOFAC_N','NEOFAC_E','PainInterf_Tscore','PainIntens_RawScore','PainInterf_Tscore','Taste_AgeAdj'\
	'Mars_Final','PSQI_Score','VSPLOT_TC']

	# to_keep = ['MMSE_Score','PicSeq_AgeAdj','CardSort_AgeAdj','Flanker_AgeAdj','PMAT24_A_CR',\
	# 'ReadEng_AgeAdj','PicVocab_AgeAdj','ProcSpeed_AgeAdj','DDisc_AUC_40K','DDisc_AUC_200',\
	# 'SCPT_SEN','SCPT_SPEC','IWRD_TOT','ListSort_AgeAdj','VSPLOT_TC',\
	# 'ER40_CR','ER40ANG','ER40FEAR','ER40HAP','ER40NOE','ER40SAD']

	for s in fin.Subject.values:
		if str(int(s)) not in subjects: fin.drop(fin[fin.Subject.values == s].index,axis=0,inplace=True)
	assert (np.array(fin.Subject.values) == np.array(subjects).astype(int)).all()
	for c in fin.columns:
		if c not in to_keep: fin = fin.drop(c,axis=1)
	for c in fin.columns:
		a = fin[c][np.isnan(fin[c])]
		assert len(a[a==True]) == 0
		fin[c][np.isnan(fin[c])] = np.nanmean(fin[c])
	return fin

def make_heatmap(data,cmap="RdBu_r",dmin=None,dmax=None):
	minflag = False
	maxflag = False
	orig_colors = sns.color_palette(cmap,n_colors=1001)
	norm_data = np.array(copy.copy(data))
	if dmin != None: 
		if dmin > np.min(norm_data):norm_data[norm_data<dmin]=dmin
		else: 
			norm_data=np.append(norm_data,dmin)
			minflag = True
	if dmax != None: 
		if dmax < np.max(norm_data):norm_data[norm_data>dmax]=dmax
		else:
			norm_data=np.append(norm_data,dmax)
			maxflag = True
	if np.nanmin(data) < 0.0: norm_data = norm_data + (np.nanmin(norm_data)*-1)
	elif np.nanmin(data) > 0.0: norm_data = norm_data - (np.nanmin(norm_data))
	norm_data = norm_data / float(np.nanmax(norm_data))
	norm_data = norm_data * 1000
	norm_data = norm_data.astype(int)
	colors = []
	for d in norm_data:
		colors.append(orig_colors[d])
	if maxflag: colors = colors[:-1]
	if minflag: colors = colors[:-1]
	return colors

def performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=False,use_matrix=True,return_df=False):
	try: del pool
	except: pass
	pool = Pool(multiprocessing.cpu_count()-1)
	run_version = 'fz'
	# control_com=False
	# control_motion=False
	# use_matrix = True
	tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL']
	atlas='power'
	loo_columns= ['Task','Predicted Performance','Performance']
	loo_df = pd.DataFrame(columns = loo_columns)
	pc_df = pd.DataFrame(columns=['Task','Mean Participation Coefficient','Diversity Facilitated Modularity Coefficient'])
	wmd_df = pd.DataFrame(columns=['Task','Mean Within-Community-Strength','Locality Facilitated Modularity Coefficient'])
	total_subs = np.array([])
	for task in tasks:
		"""
		preprocessing 
		"""
		print task.capitalize()
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
	
		all_subs = np.intersect1d(rest_subjects,subjects)
		rest_idx = []
		task_idx = []
		for s in all_subs:
			rest_idx.append(np.where(rest_subjects == s)[0][0])
			task_idx.append(np.where(subjects == s)[0][0])
		assert (rest_subjects[rest_idx] == subjects[task_idx]).all()
		subjects = all_subs
		print len(np.unique(subjects)),len(subjects)
		total_subs = np.append(total_subs,subjects.copy())
		print len(np.unique(np.array(total_subs).flatten()))
		continue

		rest_subject_pcs = rest_subject_pcs[rest_idx]
		rest_subject_wmds = rest_subject_wmds[rest_idx]
		rest_subject_mods = rest_subject_mods[rest_idx]
		rest_matrices= rest_matrices[rest_idx]
		subject_pcs = subject_pcs[task_idx]
		subject_wmds = subject_wmds[task_idx]
		subject_mods = subject_mods[task_idx]
		matrices = matrices[task_idx]
		subject_communities = subject_communities[task_idx]
		
		task_perf = task_performance(np.array(subjects).astype(int),task)
		to_delete = np.isnan(task_perf).copy()
		to_delete = np.where(to_delete==True)
		task_perf = np.delete(task_perf,to_delete)
		subjects = np.delete(subjects,to_delete)
		if control_motion == True:
			subject_motion = []
			for subject in subjects:
				subject_motion.append(get_sub_motion(subject,task))
			assert np.min(subject_motion) > 0.0
			subject_motion = np.array(subject_motion)
		subject_pcs = np.delete(subject_pcs,to_delete,axis=0)
		subject_mods = np.delete(subject_mods,to_delete)
		subject_wmds = np.delete(subject_wmds,to_delete,axis=0)
		matrices = np.delete(matrices,to_delete,axis=0)
		subject_communities = np.delete(subject_communities,to_delete)
		
		rest_subject_pcs = np.delete(rest_subject_pcs,to_delete,axis=0)
		rest_subject_wmds = np.delete(rest_subject_wmds,to_delete,axis=0)
		rest_subject_mods = np.delete(rest_subject_mods,to_delete,axis=0)
		rest_matrices = np.delete(rest_matrices,to_delete,axis=0)

		subject_pcs[np.isnan(subject_pcs)] = 0.0
		rest_subject_pcs[np.isnan(rest_subject_pcs)] = 0.0
		rest_subject_wmds[np.isnan(rest_subject_wmds)] = 0.0
		subject_wmds[np.isnan(subject_wmds)] = 0.0

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

		assert subject_pcs.shape[0] == len(subjects)
		
		task_pc_corr = np.zeros(subject_pcs.shape[1])
		for i in range(len(task_pc_corr)):
			task_pc_corr[i] = nan_pearsonr(task_perf,subject_pcs[:,i])[0]
		task_wmd_corr = np.zeros(subject_pcs.shape[1])
		for i in range(len(task_pc_corr)):
			task_wmd_corr[i] = nan_pearsonr(task_perf,subject_wmds[:,i])[0]
		task_str = np.zeros((len(task_pc_corr))).astype(str)
		task_str[:] = task.capitalize()
		pc_df = pc_df.append(pd.DataFrame(np.array([task_str,np.nanmean(subject_pcs,axis=0),task_pc_corr]).transpose(),columns=['Task','Mean Participation Coefficient','Diversity Facilitated Performance Coefficient']),ignore_index=True)
		wmd_df = wmd_df.append(pd.DataFrame(np.array([task_str,np.nanmean(subject_wmds,axis=0),task_wmd_corr]).transpose(),columns=['Task','Mean Within-Community-Strength','Locality Facilitated Performance Coefficient']),ignore_index=True)
		if return_df:
			continue
		"""
		prediction / cross validation
		"""
		vs = []
		for t in range(len(task_perf)):
			vs.append([subject_pcs,subject_wmds,subject_mods,rest_subject_pcs,rest_subject_wmds,rest_subject_mods,task_perf,t,matrices,rest_matrices,False,use_matrix])
		nodal_prediction = pool.map(super_edge_predict_new,vs)
		result = pearsonr(np.array(nodal_prediction).reshape(-1),task_perf)
		print 'Prediction of Performance: ', result
		sys.stdout.flush()
		loo_array = []
		for i in range(len(nodal_prediction)):
			loo_array.append([task,nodal_prediction[i],task_perf[i]])
		loo_df = pd.concat([loo_df,pd.DataFrame(loo_array,columns=loo_columns)],axis=0)
	# plot_results(loo_df,'Predicted Performance','Performance','/home/despoB/mb3152/dynamic_mod/figures/Predicted_Performance_%s_%s_%s.pdf'%(run_version,control_motion,use_matrix))
	if return_df:return pc_df,wmd_df
	
def performance_across_traits(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=False):
	try: del pool
	except: pass
	pool = Pool(multiprocessing.cpu_count()-1)	
	atlas = 'power'
	run_version = 'fz'
	control_com=False
	control_motion=False
	tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL']
	project='hcp'
	atlas='power'

	behavior_df = pd.DataFrame(columns=['Task','Behavioral Measure','Prediction Accuracy','p'])
	prediction_df = pd.DataFrame(columns=['Task','Behavioral Measure','Prediction Accuracy','p'])
	
	for task in tasks:
		"""
		preprocessing 
		"""
		print task.capitalize()
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

	
		all_subs = np.intersect1d(rest_subjects,subjects)
		rest_idx = []
		task_idx = []
		for s in all_subs:
			rest_idx.append(np.where(rest_subjects == s)[0][0])
			task_idx.append(np.where(subjects == s)[0][0])

		assert (rest_subjects[rest_idx] == subjects[task_idx]).all()
		
		subjects = all_subs

		rest_subject_pcs = rest_subject_pcs[rest_idx]
		rest_subject_wmds = rest_subject_wmds[rest_idx]
		rest_subject_mods = rest_subject_mods[rest_idx]
		rest_matrices= rest_matrices[rest_idx]
		subject_pcs = subject_pcs[task_idx]
		subject_wmds = subject_wmds[task_idx]
		subject_mods = subject_mods[task_idx]
		matrices = matrices[task_idx]
		subject_communities = subject_communities[task_idx]
		
		task_perf = task_performance(np.array(subjects).astype(int),task)
		to_delete = np.isnan(task_perf).copy()
		to_delete = np.where(to_delete==True)
		task_perf = np.delete(task_perf,to_delete)
		subjects = np.delete(subjects,to_delete)
		if control_motion == True:
			subject_motion = []
			for subject in subjects:
				subject_motion.append(get_sub_motion(subject,task))
			assert np.min(subject_motion) > 0.0
			subject_motion = np.array(subject_motion)
		subject_pcs = np.delete(subject_pcs,to_delete,axis=0)
		subject_mods = np.delete(subject_mods,to_delete)
		subject_wmds = np.delete(subject_wmds,to_delete,axis=0)
		matrices = np.delete(matrices,to_delete,axis=0)
		subject_communities = np.delete(subject_communities,to_delete)
		
		rest_subject_pcs = np.delete(rest_subject_pcs,to_delete,axis=0)
		rest_subject_wmds = np.delete(rest_subject_wmds,to_delete,axis=0)
		rest_subject_mods = np.delete(rest_subject_mods,to_delete,axis=0)
		rest_matrices = np.delete(rest_matrices,to_delete,axis=0)

		subject_pcs[np.isnan(subject_pcs)] = 0.0
		rest_subject_pcs[np.isnan(rest_subject_pcs)] = 0.0
		subject_wmds[np.isnan(subject_wmds)] = 0.0

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

		assert subject_pcs.shape[0] == len(subjects)
		
		"""
		generalize features to behavioral measures
		"""
		fin = behavior(subjects)
		continue
		translation = {'Loneliness_Unadj': 'Lonelisness','PercReject_Unadj':'Percieved Rejection','AngHostil_Unadj':'Hostility','Sadness_Unadj':'Sadness','PercHostil_Unadj':'Percieved Hostility','NEOFAC_N':'Neuroticism',\
		'FearAffect_Unadj':'Fear','AngAggr_Unadj':'Agressive Anger','PainInterf_Tscore':'Pain Interferes With Daily Life','Strength_AgeAdj':'Physical Strength','FearSomat_Unadj':'Somatic Fear','PSQI_Score':'Poor Sleep',\
		'SCPT_SPEC':'Sustained Attention Specificity','SCPT_SEN':'Sustained Attention Sensativity','ER40HAP':'Emotion, Happy Identifications','DDisc_AUC_200':'Delay Discounting:$200',\
		'GaitSpeed_Comp':'Gait Speed','DDisc_AUC_40K':'Delay Discounting: $40,000','ER40NOE':'Emotion, Neutral Identifications','ER40ANG':'Emotion, Angry Identifications',\
		'ER40FEAR':'Emotion, Fearful Identifications','ER40SAD':'Emotion, Sad Identifications','ER40_CR':'Emotion Recognition','MMSE_Score':'Mini Mental Status Exam','NEOFAC_O':'Openness','IWRD_TOT':'Verbal Memory','PMAT24_A_CR':'Penn Matrix','NEOFAC_C':'Conscientiousness',\
		'NEOFAC_A':'Agreeableness','Flanker_AgeAdj':'Flanker Task','CardSort_AgeAdj':'Card Sorting Task','NEOFAC_E':'Extraversion','Dexterity_AgeAdj':'Dexterity','Endurance_AgeAdj':'Endurance','ReadEng_AgeAdj':'Oral Reading Recognition',\
		'PicVocab_AgeAdj':'Picture Vocabulary','ProcSpeed_AgeAdj':'Processing Speed','SelfEff_Unadj':'Percieved Stress','PosAffect_Unadj':'Positive Affect','MeanPurp_Unadj':'Meaning and Purpose','Friendship_Unadj':'Friendship',\
		'PicSeq_AgeAdj':'Picture Sequence Memory','LifeSatisf_Unadj':'Life Satisfaction','EmotSupp_Unadj':'Emotional Support','ListSort_AgeAdj':'Working Memory','VSPLOT_TC':'Spatial','AngAffect_Unadj':'Anger, Affect'}
		vs = []
		for t in range(len(task_perf)):
			vs.append([subject_pcs,subject_wmds,subject_mods,rest_subject_pcs,rest_subject_wmds,rest_subject_mods,task_perf,t,matrices,rest_matrices,True,True])
		task_model_results = pool.map(super_edge_predict_new,vs)
		task_model = []
		for idx in range(len(task_model_results)):
			task_model.append(task_model_results[idx][:-1])
		task_model = np.array(task_model)
		for i in range(fin.shape[1]):
			behav_perf = fin[fin.columns.values[i]]
			vs = []
			for t in range(len(task_perf)):
				vs.append([subject_pcs,subject_wmds,subject_mods,rest_subject_pcs,rest_subject_wmds,rest_subject_mods,behav_perf,t,matrices,rest_matrices,True,True])
			behav_model_results = pool.map(super_edge_predict_new,vs)
			behav_model = []
			behav_prediction = []
			for idx in range(len(behav_model_results)):
				behav_model.append(behav_model_results[idx][:-1])
				behav_prediction.append(behav_model_results[idx][-1])
			behav_model = np.array(behav_model)
			behav_prediction = np.array(behav_prediction)
			fits = []
			pvals = []
			for feat in range(behav_model.shape[2]):
				fits.append(pearsonr(np.array(behav_model)[:,0,feat],np.array(task_model)[:,0,feat])[0])
				pvals.append(pearsonr(np.array(behav_model)[:,0,feat],np.array(task_model)[:,0,feat])[1])
			behavior_df = behavior_df.append(pd.DataFrame(np.array([task,translation[fin.columns.values[i]],np.mean(fits[:-2]),np.mean(pvals[:-2])]).reshape(1,4),columns=['Task','Behavioral Measure','Prediction Accuracy','p']),ignore_index=True)
			result,p = pearsonr(behav_prediction,behav_perf)
			prediction_df = prediction_df.append(pd.DataFrame(np.array([task,translation[fin.columns.values[i]],result,p]).reshape(1,4),columns=['Task','Behavioral Measure','Prediction Accuracy','p']),ignore_index=True)
			behavior_df.to_csv('/home/despoB/mb3152/dynamic_mod/feature_corr.csv')
			prediction_df.to_csv('/home/despoB/mb3152/dynamic_mod/feature_behav_predict.csv')
		
		# plot(behavior_df,savestr,colormap='coolwarm')
		# plot(prediction_df,savestr,colormap='coolwarm')
	# tasks = ['LANGUAGE','RELATIONAL','SOCIAL','WM']		
	# m = np.zeros((4,4))
	# for i,t1 in enumerate(tasks):
	# 	for j,t2 in enumerate(tasks):
	# 		m[i,j] = pearsonr(df['Prediction Accuracy'][df.Task==t1],df['Prediction Accuracy'][df.Task==t2])[0]
	# 		m[j,i] = pearsonr(df['Prediction Accuracy'][df.Task==t1],df['Prediction Accuracy'][df.Task==t2])[0]
	# tasks = ['Language','Relational','Social','Working Memory']	
	# np.fill_diagonal(m,np.nan)
	# sns.heatmap(m)
	# sns.plt.xticks(range(4),tasks,rotation=90)
	# tasks.reverse()
	# sns.plt.yticks(range(4),tasks,rotation=360)
	# sns.plt.tight_layout()
	# sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/feature_corr_corr.pdf')
	# sns.plt.show()

def plot(df,savestr,colormap='coolwarm'):
	behavior_df = df
	tasks = np.unique(behavior_df.Task.values)
	behavior_df['Prediction Accuracy'] = behavior_df['Prediction Accuracy'].values.astype(float)
	behavior_df['colors'] = make_heatmap(behavior_df['Prediction Accuracy'].values,colormap,-.3,.3)
	
	norm_behavior_df = behavior_df.copy()
	for task in behavior_df['Task'].values:
		norm_behavior_df['Prediction Accuracy'][norm_behavior_df['Task']==task] = scipy.stats.zscore(norm_behavior_df['Prediction Accuracy'][norm_behavior_df['Task']==task].values)
	for measure in behavior_df['Behavioral Measure'].values:
		norm_behavior_df['Prediction Accuracy'][norm_behavior_df['Behavioral Measure']==measure] = scipy.stats.zscore(norm_behavior_df['Prediction Accuracy'][norm_behavior_df['Behavioral Measure']==measure].values)
	
	norm_behavior_df['colors'] = make_heatmap(norm_behavior_df['Prediction Accuracy'].values,colormap,-1.5,1.5)
	
	left, width = 0, 1
	bottom, height = 0, 1
	right = left + width
	top = bottom + height
	top = top /2.
	fig = plt.figure(figsize=(mm_2_inches(183),mm_2_inches(247)))
	for col,task in zip(np.linspace(.135,.865,4),tasks):
		order = behavior_df[behavior_df.Task==task]['Behavioral Measure'].values[np.argsort(behavior_df[behavior_df.Task==task]['Prediction Accuracy'].values)]
		scores = behavior_df[behavior_df.Task==task]['Prediction Accuracy'].values[np.argsort(behavior_df[behavior_df.Task==task]['Prediction Accuracy'].values)]
		pvals = behavior_df[behavior_df.Task==task]['p'].values[np.argsort(behavior_df[behavior_df.Task==task]['Prediction Accuracy'].values)].astype(float)
		for ix,o in enumerate(order):
			if float(scores[ix]) < 0.0: 
				s = '-' + str(scores[ix])[1:5]
				order[ix] = order[ix] + ' (%s)'%(s)
				continue
			s = str(scores[ix])[1:4]
			order[ix] = order[ix] + ' (%s)'%(s)
			
		order = np.append(order,task.capitalize())
		colors = norm_behavior_df[norm_behavior_df.Task==task]['colors'].values[np.argsort(norm_behavior_df[norm_behavior_df.Task==task]['Prediction Accuracy'].values)]
		# colors = norm_behavior_df[norm_behavior_df.Task==task]['colors'].values[np.argsort(norm_behavior_df[norm_behavior_df.Task==task]['Prediction Accuracy'].values)]
		colors = list(colors)
		colors.append((0,0,0))
		pvals = np.append(pvals,1)
		locs = (np.arange(len(order)+1)/float(len(order)+1))[1:]
		for i,t,c,p in zip(locs,order,colors,pvals):
			if t == 'Wm': t = 'Working Memory'
			if p <  (.05 / len(colors)):
				t = t + " *"
			fig.text(col*(left+right), float(i)*(bottom+top), t,horizontalalignment='center',verticalalignment='center',fontsize=7, color=c)
	sns.plt.savefig('%s.pdf'%(savestr))
	sns.plt.show()

def small_tstatfunc(x, y,bc=False):
	t, p = scipy.stats.ttest_ind(x,y)

	if p < 1e-5: pst = '*!'
	elif p < .05: pst = '*'
	else: pst = None
	if pst == None: return "%s" %(np.around(t,3))
	else: return "%s \n p%s" %(np.around(t,3),p)

def plot_box(data,x,y,split_names,savestr,colors):
	data['Node Type'] = np.zeros(len(data)).astype(str)
	data.Task = data.Task.str.capitalize()
	for task in data.Task.values:
		metric= data[x][data.Task==task].values
		cut_off = np.percentile(metric,80)
		c = np.zeros(len(metric)).astype(str)
		c[metric >= cut_off] = split_names[0]
		c[metric < cut_off] = split_names[1]
		data['Node Type'][data.Task==task] = c
	sns.set(context="paper",font='Helvetica',style="white",font_scale=1.5)
	ax = sns.boxplot(data=data,x='Task',y=y,hue='Node Type',hue_order=split_names,palette=colors)
	for t in np.unique(data.Task.values):
		tdf = data[data.Task==t]
		print t, scipy.stats.ttest_ind(tdf[y][tdf['Node Type']==split_names[0]].values,tdf[y][tdf['Node Type']==split_names[1]].values)
	# 	maxvaly = np.mean(tdf[y]) + (np.std(tdf[y]) * .5)
	# 	sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.tight_layout()
	sns.plt.savefig(savestr)
	sns.plt.close()
# performance_across_tasks()
# connectivity_across_tasks()
# plot(pd.read_csv('/home/despoB/mb3152/dynamic_mod/feature_behav_predict.csv'),'feature_behav_predict','Reds')
# plot(pd.read_csv('/home/despoB/mb3152/dynamic_mod/feature_corr.csv'),'feature_corr','RdBu_r')
# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='scrub_.2',control_com=False,control_motion=False,use_matrix=False)
# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='scrub_.2',control_com=False,control_motion=False,use_matrix=True)

# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=False,use_matrix=False)
# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=False,use_matrix=True)

# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=True,use_matrix=False)
# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=True,use_matrix=True)
# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=True,control_motion=False,use_matrix=True)
# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=True,control_motion=False,use_matrix=False)


# performance_across_traits()

# qsub -pe threaded 20 -binding linear:20  -V -l mem_free=20G -j y -o /home/despoB/mb3152/dynamic_mod/sge/ -e /home/despoB/mb3152/dynamic_mod/sge/ -N 'pred' hcp_perf2.py

"""
PRETTY FIGURES 
"""
# loo_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/loo_df.csv')
# df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/df.csv')
# plot_results(loo_df,'Predicted Performance','Performance','/home/despoB/mb3152/dynamic_mod/figures/Predicted_Performance_%s_%s.pdf'%(run_version,control_motion))
# plot_results(df,'PCxPerformance','PC','/home/despoB/mb3152/dynamic_mod/figures/PC_PC_Performance.pdf')
# plot_results(df,'PCxPerformance','PCxModularity','/home/despoB/mb3152/dynamic_mod/figures/PC_Modularity_PC_Performance.pdf')
# plot_results(df,'WCDxPerformance','WCD','/home/despoB/mb3152/dynamic_mod/figures/WCD_WCD_Performance.pdf')
# plot_results(df,'WCDxPerformance','WCDxModularity','/home/despoB/mb3152/dynamic_mod/figures/WCD_Modularity_PC_Performance.pdf')

# pc_df,wmd_df = connectivity_across_tasks(atlas='power',project='hcp',tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST'],run_version='fz_wc',control_com=False,control_motion=False)
# pc_df['Mean Participation Coefficient'] = pc_df['Mean Participation Coefficient'].astype(float)
# pc_df['Diversity Facilitated Modularity Coefficient'] = pc_df['Diversity Facilitated Modularity Coefficient'].astype(float)
# colors = np.array(sns.color_palette("cubehelix", 8))[np.array([6,7])]
# plot_box(pc_df,'Mean Participation Coefficient','Diversity Facilitated Modularity Coefficient',['Connector Hub','Other Node'],savestr='dfmc_cutoff.pdf',colors=colors)

# wmd_df['Mean Within-Community-Strength'] = wmd_df['Mean Within-Community-Strength'].astype(float)
# wmd_df['Locality Facilitated Modularity Coefficient'] = wmd_df['Locality Facilitated Modularity Coefficient'].astype(float)
# colors = np.array(sns.color_palette("cubehelix", 8))[np.array([5,7])]
# plot_box(wmd_df,'Mean Within-Community-Strength','Locality Facilitated Modularity Coefficient',['Local Hub','Other Node'],savestr='lfmc_cutoff.pdf',colors=colors)

# pc_df,wmd_df = performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=False,use_matrix=True,return_df=True)
# pc_df['Mean Participation Coefficient'] = pc_df['Mean Participation Coefficient'].astype(float)
# pc_df['Diversity Facilitated Performance Coefficient'] = pc_df['Diversity Facilitated Performance Coefficient'].astype(float)
# colors = np.array(sns.color_palette("cubehelix", 8))[np.array([6,7])]
# plot_box(pc_df,'Mean Participation Coefficient','Diversity Facilitated Performance Coefficient',['Connector Hub','Other Node'],savestr='dfpc_cutoff.pdf',colors=colors)

# wmd_df['Mean Within-Community-Strength'] = wmd_df['Mean Within-Community-Strength'].astype(float)
# wmd_df['Locality Facilitated Performance Coefficient'] = wmd_df['Locality Facilitated Performance Coefficient'].astype(float)
# colors = np.array(sns.color_palette("cubehelix", 8))[np.array([5,7])]
# plot_box(wmd_df,'Mean Within-Community-Strength','Locality Facilitated Performance Coefficient',['Local Hub','Other Node'],savestr='lfpc_cutoff.pdf',colors=colors)

# plot_connectivity_results(pc_df,'Diversity Facilitated Modularity Coefficient','Mean Participation Coefficient','/home/despoB/mb3152/dynamic_mod/figures/pc_pc_w_c.pdf')

# wmd_df['Mean Within-Community-Strength'] = wmd_df['Mean Within-Community-Strength'].astype(float)
# wmd_df['Locality Facilitated Modularity Coefficient'] = wmd_df['Locality Facilitated Modularity Coefficient'].astype(float)
# plot_connectivity_results(wmd_df,'Locality Facilitated Modularity Coefficient','Mean Within-Community-Strength','/home/despoB/mb3152/dynamic_mod/figures/wmd_wmd_q_c.pdf')

# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=False)
# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='fz',control_com=False,control_motion=True)
# performance_across_tasks(atlas='power',tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL'],run_version='scrub_.2',control_com=False,control_motion=False)
# performance_across_tasks()

# performance_across_traits()


# if len(sys.argv) > 1:
# 	if sys.argv[1] == 'perf':
# 		performance_across_tasks()
# 	if sys.argv[1] == 'forever':
# 		a = 0
# 		while True:
# 			a = a - 1
# 			a = a + 1
# 	if sys.argv[1] == 'pc_edge_corr':
# 		task = sys.argv[2]
# 		atlas = 'power'
# 		subjects = np.array(hcp_subjects).copy()
# 		subjects = list(subjects)
# 		subjects = remove_missing_subjects(subjects,task,atlas)
# 		static_results = graph_metrics(subjects,task,atlas)
# 		subject_pcs = static_results['subject_pcs']
# 		matrices = static_results['matrices']
# 		pc_edge_corr = pc_edge_correlation(subject_pcs,matrices,path='/home/despoB/mb3152/dynamic_mod/results/hcp_%s_power_pc_edge_corr_z.npy' %(task))
# 	if sys.argv[1] == 'graph_metrics':
# 		# subjects = remove_missing_subjects(list(np.array(hcp_subjects).copy()),sys.argv[2],sys.argv[3])
# 		# subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',sys.argv[2],sys.argv[3]))
# 		# graph_metrics(subjects,task=sys.argv[2],atlas=sys.argv[3],run_version='fz_wc',run=True)
# 		subjects = []
# 		dirs = os.listdir('/home/despoB/connectome-data/')
# 		for s in dirs:
# 			try: int(s)
# 			except: continue
# 			subjects.append(str(s))
# 		graph_metrics(subjects,task=sys.argv[2],atlas=sys.argv[3],run_version='HCP_900',run=True)
# 	if sys.argv[1] == 'make_matrix':
# 		subject = str(sys.argv[2])
# 		task = str(sys.argv[3])
# 		atlas = str(sys.argv[4])
# 		make_static_matrix(subject,task,'hcp',atlas)
# 	if sys.argv[1] == 'calc_motion':
# 		subject = str(sys.argv[2])
# 		task = str(sys.argv[3])
# 		run_fd(subject,task)
# 	if sys.argv[1] == 'check_norm':
# 		atlas = sys.argv[3]
# 		task = sys.argv[2]
# 		# subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',sys.argv[2],sys.argv[3]))
# 		subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_scrub_.2.npy' %('hcp',sys.argv[2],sys.argv[3]))
# 		check_scrubbed_normalize(subjects,task,atlas='power')
# 		print 'done checkin, all good!'
# 	if sys.argv[1] == 'mediation':
# 		local_mediation(sys.argv[2]) 
# 	if sys.argv[1] == 'alg_compare':
# 		subjects = []
# 		dirs = os.listdir('/home/despoB/connectome-data/')
# 		for s in dirs:
# 			try: int(s)
# 			except: continue
# 			subjects.append(str(s))
# 	# 	alg_compare(subjects)

