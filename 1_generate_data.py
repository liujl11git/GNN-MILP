# This script generates random MILP instances for training and testing

import numpy as np
import random as rd
import os
import argparse
from pandas import read_csv 
import pyscipopt as scip

## ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--exp_env",    default='2',	choices = ['1','2'])
parser.add_argument("--m",          default='6')
parser.add_argument("--n",          default='20')
parser.add_argument("--nnz",        default='60')
args = parser.parse_args()

## SETUP
m = int(args.m)                     # number of constraints
n = int(args.n)                     # number of variables
nnz = int(args.nnz)                 # number of nonzero elements in A 
exp_env = int(args.exp_env)			# experiments environment (1 means foldable vs unfoldable; 2 means training vs testing)
rd.seed(0)

## DATA GENERATION
def generateMILPunfoldable(k_data, configs, folder):
	'''
	This function generates and saves unfoldable MILP instances.
	- k_data: the number of instances you want to generate 
	- configs: (m,n,nnz), configurations of each MILP instance 
	- folder: the folder you want to save those generated MILPs
	'''
	m,n,nnz,env = configs
	
	count_feas = 0

	for k in range(k_data):
		path = folder + "MIP_env" + str(env) + "_unfoldable_" + str(k)
		if not os.path.exists(path):
			os.makedirs(path)

		c = np.array([ rd.normalvariate(0, 1) for j in range(n) ]) * 0.01
		b = np.array([ rd.normalvariate(0, 1) for i in range(m) ])
		circ = np.array([ rd.randint(0, 2) for i in range(m) ])
		NI = np.array([ rd.randint(0, 1) for j in range(n) ])
		lb = np.array([ rd.normalvariate(0, 10) for j in range(n) ])
		ub = np.array([ rd.normalvariate(0, 10) for j in range(n) ])

		for j in range(n):
			if lb[j] > ub[j]:
				temp = lb[j]
				lb[j] = ub[j]
				ub[j] = temp

		A = np.zeros((m, n))
		EdgeIndex = np.zeros((nnz, 2))
		EdgeIndex1D = rd.sample(range(m * n), nnz)
		EdgeFeature = np.array([ rd.normalvariate(0, 1) for l in range(nnz) ])
		
		for l in range(nnz):
			i = int(EdgeIndex1D[l] / n)
			j = EdgeIndex1D[l] - i * n
			EdgeIndex[l, 0] = i
			EdgeIndex[l, 1] = j
			A[i, j] = EdgeFeature[l]

		opt_model = scip.Model("Unfoldable MIP Model " + str(k))
		opt_model.hideOutput()

		x_vars = []

		for j in range(n):
			if NI[j] == 1:
				x_vars.append( opt_model.addVar(vtype = "INTEGER", lb = lb[j], ub = ub[j], name="x_{0}".format(j)) )
			else:
				x_vars.append( opt_model.addVar(vtype = "CONTINUOUS", lb = lb[j], ub = ub[j], name="x_{0}".format(j)) )

		for i in range(m):
			if circ[i] == 0:
				opt_model.addCons(scip.quicksum(A[i,j] * x_vars[j] for j in range(n)) <= b[i], name="constraint_{0}".format(i))
			elif circ[i] == 1:
				opt_model.addCons(scip.quicksum(A[i,j] * x_vars[j] for j in range(n)) == b[i], name="constraint_{0}".format(i))
			else:
				opt_model.addCons(scip.quicksum(A[i,j] * x_vars[j] for j in range(n)) >= b[i], name="constraint_{0}".format(i))

		opt_model.setObjective(scip.quicksum(x_vars[j] * c[j] for j in range(n)), "minimize")
		
		## save the MILP instance
		# opt_model.writeProblem(filename = path + ".mps")
		np.savetxt(path + '/ConFeatures.csv', np.hstack((b.reshape(m, 1), circ.reshape(m, 1))), delimiter = ',', fmt = '%10.5f')
		np.savetxt(path + '/EdgeFeatures.csv', EdgeFeature, fmt = '%10.5f')
		np.savetxt(path + '/EdgeIndices.csv', EdgeIndex, delimiter = ',', fmt = '%d')
		np.savetxt(path + '/VarFeatures.csv', np.hstack((c.reshape(n, 1), NI.reshape(n, 1), lb.reshape(n, 1), ub.reshape(n, 1))), delimiter = ',', fmt = '%10.5f')
		
		## call SCIP solver to solve the MILP
		opt_model.optimize()
		
		## save the results
		if opt_model.getStatus() == "optimal":
			count_feas += 1
			sol = np.array([opt_model.getVal(x) for x in x_vars]).T
			obj = opt_model.getObjVal()
			np.savetxt(path + '/Labels_feas.csv', [1], fmt = '%d')
			np.savetxt(path + '/Labels_obj.csv', [obj], fmt = '%10.5f')
			np.savetxt(path + '/Labels_solu.csv', sol, fmt = '%10.5f')
			print("MILP generated:", k, '/', k_data, "Status: optimal.")
		elif opt_model.getStatus() == "infeasible":
			np.savetxt(path + '/Labels_feas.csv', [0], fmt = '%d')
			print("MILP generated:", k, '/', k_data, "Status: infeasible.")
		else:
			print("Unexpected model status. Quit.")
			quit()

	print("Ratio of feasible instances:", count_feas, '/', k_data)
	

def generateMILPfoldable(k_data, configs, folder):
	'''
	This function generates and saves foldable MILP instances.
	- k_data: the number of instances you want to generate 
	- configs: (m,n), configurations of each LP instance 
	- folder: the folder you want to save those generated LPs
	'''
	m,n,nnz,env = configs
	
	count_feas = 0

	for k in range(int(k_data / 2)):
		k1 = 2 * k
		path1 = folder + "MIP_env" + str(env) + "_foldable_" + str(k1)
		if not os.path.exists(path1):
			os.makedirs(path1)

		k2 = 2 * k + 1
		path2 = folder + "MIP_env" + str(env) + "_foldable_" + str(k2)
		if not os.path.exists(path2):
			os.makedirs(path2)

		if env == 2:
			c = np.ones((n,)) * 0.01
		else:
			c = np.zeros((n,))
		b = np.array([ 1 for i in range(m) ])
		
		lb = np.array([ rd.normalvariate(0, 10) for j in range(n) ])
		ub = np.array([ rd.normalvariate(0, 10) for j in range(n) ])
		NI = np.array([ 0 for j in range(n) ])

		for j in range(n):
			if lb[j] > ub[j]:
				temp = lb[j]
				lb[j] = ub[j]
				ub[j] = temp

		j_list = rd.sample(range(n), 6)
		for j in j_list:
			lb[j] = 0
			ub[j] = 1
			NI[j] = 1

		A1 = np.zeros((m, n))
		EdgeIndex1 = np.zeros((12, 2))
		EdgeFeature1 = np.ones((12, 1))
		for i in range(6):
			j1 = j_list[i]
			if i == 5:
				j2 = j_list[0]
			else:
				j2 = j_list[i + 1]
			
			EdgeIndex1[2 * i, 0] = i
			EdgeIndex1[2 * i + 1, 0] = i
			EdgeIndex1[2 * i, 1] = j1
			EdgeIndex1[2 * i + 1, 1] = j2
		
		A2 = np.zeros((m, n))
		EdgeIndex2 = np.zeros((12, 2))
		EdgeFeature2 = np.ones((12, 1))
		for i in range(6):
			j1 = j_list[i]
			if i == 2:
				j2 = j_list[0]
			elif i == 5:
				j2 = j_list[3]
			else:
				j2 = j_list[i + 1]
			
			EdgeIndex2[2 * i, 0] = i
			EdgeIndex2[2 * i + 1, 0] = i
			EdgeIndex2[2 * i, 1] = j1
			EdgeIndex2[2 * i + 1, 1] = j2
		
		for l in range(12):
			i = int(EdgeIndex1[l, 0])
			j = int(EdgeIndex1[l, 1])
			A1[i, j] = EdgeFeature1[l, 0]
			i = int(EdgeIndex2[l, 0])
			j = int(EdgeIndex2[l, 1])
			A2[i, j] = EdgeFeature2[l, 0]

		opt_model1 = scip.Model("Foldable MIP Model" + str(k1))
		opt_model2 = scip.Model("Foldable MIP Model" + str(k2))
		opt_model1.hideOutput()
		opt_model2.hideOutput()

		x_vars1 = []
		x_vars2 = []

		for j in range(n):
			if NI[j] == 1:
				x_vars1.append( opt_model1.addVar(vtype = "INTEGER", lb = lb[j], ub = ub[j], name="x_{0}".format(j)) )
				x_vars2.append( opt_model2.addVar(vtype = "INTEGER", lb = lb[j], ub = ub[j], name="x_{0}".format(j)) )
			else:
				x_vars1.append( opt_model1.addVar(vtype = "CONTINUOUS", lb = lb[j], ub = ub[j], name="x_{0}".format(j)) )
				x_vars2.append( opt_model2.addVar(vtype = "CONTINUOUS", lb = lb[j], ub = ub[j], name="x_{0}".format(j)) )

		for i in range(m):
			opt_model1.addCons(scip.quicksum(A1[i,j] * x_vars1[j] for j in range(n)) == b[i], name="constraint_{0}".format(i))
			opt_model2.addCons(scip.quicksum(A2[i,j] * x_vars2[j] for j in range(n)) == b[i], name="constraint_{0}".format(i))

		opt_model1.setObjective(scip.quicksum(x_vars1[j] * c[j] for j in range(n)), "minimize")
		opt_model2.setObjective(scip.quicksum(x_vars2[j] * c[j] for j in range(n)), "minimize")
		
		## save the MILP instance
		# opt_model1.writeProblem(filename = path1 + ".mps")
		# opt_model2.writeProblem(filename = path2 + ".mps")
		np.savetxt(path1 + '/ConFeatures.csv', np.hstack((b.reshape(m, 1), np.ones((m, 1)))), delimiter = ',', fmt = '%10.5f')
		np.savetxt(path1 + '/EdgeFeatures.csv', EdgeFeature1, fmt = '%10.5f')
		np.savetxt(path1 + '/EdgeIndices.csv', EdgeIndex1, delimiter = ',', fmt = '%d')
		np.savetxt(path1 + '/VarFeatures.csv', np.hstack((c.reshape(n, 1), NI.reshape(n, 1), lb.reshape(n, 1), ub.reshape(n, 1))), delimiter = ',', fmt = '%10.5f')
		np.savetxt(path2 + '/ConFeatures.csv', np.hstack((b.reshape(m, 1), np.ones((m, 1)))), delimiter = ',', fmt = '%10.5f')
		np.savetxt(path2 + '/EdgeFeatures.csv', EdgeFeature2, fmt = '%10.5f')
		np.savetxt(path2 + '/EdgeIndices.csv', EdgeIndex2, delimiter = ',', fmt = '%d')
		np.savetxt(path2 + '/VarFeatures.csv', np.hstack((c.reshape(n, 1), NI.reshape(n, 1), lb.reshape(n, 1), ub.reshape(n, 1))), delimiter = ',', fmt = '%10.5f')

		## solve MILPs
		opt_model1.optimize()
		opt_model2.optimize()

		if opt_model1.getStatus() == "optimal":
			count_feas += 1
			sol1 = np.array([opt_model1.getVal(x) for x in x_vars1]).T
			np.savetxt(path1 + '/Labels_feas.csv', [1], fmt = '%d')
			np.savetxt(path1 + '/Labels_obj.csv', [opt_model1.getObjVal()], fmt = '%10.5f')
			np.savetxt(path1 + '/Labels_solu.csv', sol1, fmt = '%10.5f')
			print("MILP generated:", k1, '/', k_data, "Status: optimal.")
		elif opt_model1.getStatus() == "infeasible":
			np.savetxt(path1 + '/Labels_feas.csv', [0], fmt = '%d')
			print("MILP generated:", k1, '/', k_data, "Status: infeasible.")
		else:
			print("Unexpected model status. Quit.")
			quit()
		
		if opt_model2.getStatus() == "optimal":
			count_feas += 1
			sol2 = np.array([opt_model2.getVal(x) for x in x_vars2]).T
			np.savetxt(path2 + '/Labels_feas.csv', [1], fmt = '%d')
			np.savetxt(path2 + '/Labels_obj.csv', [opt_model2.getObjVal()], fmt = '%10.5f')
			np.savetxt(path2 + '/Labels_solu.csv', sol2, fmt = '%10.5f')
			print("MILP generated:", k2, '/', k_data, "Status: optimal.")
		elif opt_model2.getStatus() == "infeasible":
			np.savetxt(path2 + '/Labels_feas.csv', [0], fmt = '%d')
			print("MILP generated:", k2, '/', k_data, "Status: infeasible.")
		else:
			print("Unexpected model status. Quit.")
			quit()

	print("Ratio of feasible instances:", count_feas, '/', k_data)

def combineGraphsAll(ID_start, ID_end, configs, folder):
	'''
	This function combines all MILP instances with "MIP_key" to a large graph to facilitate training.
	This function also makes labels for the feasibility of all MILP instances 
	'''
	m,n,MIP_key,is_append_rand = configs

	def my_stack(ori,aug):
		return (np.copy(aug) if ori is None else np.concatenate((ori,aug),axis=0) )

	startMIPidx = 0
	ConFeatures_all = None
	EdgeFeatures_all = None
	EdgeIndices_all = None
	VarFeatures_all = None
	Labels_feas = None

	for k in range(ID_start, ID_end):
		MIP_path = MIP_key + '_' + str(k)
		varFeatures = read_csv(MIP_path + "/VarFeatures.csv", header=None).values
		conFeatures = read_csv(MIP_path + "/ConFeatures.csv", header=None).values
		edgeFeatures = read_csv(MIP_path + "/EdgeFeatures.csv", header=None).values
		edgeIndices = read_csv(MIP_path + "/EdgeIndices.csv", header=None).values
		labelsFeas = read_csv(MIP_path + "/Labels_feas.csv", header=None).values
		
		edgeIndices[:, 0] = edgeIndices[:, 0] + startMIPidx * m
		edgeIndices[:, 1] = edgeIndices[:, 1] + startMIPidx * n
		
		ConFeatures_all = my_stack(ConFeatures_all, conFeatures)
		VarFeatures_all = my_stack(VarFeatures_all, varFeatures)
		EdgeFeatures_all = my_stack(EdgeFeatures_all, edgeFeatures)
		EdgeIndices_all = my_stack(EdgeIndices_all, edgeIndices)
		Labels_feas = my_stack(Labels_feas, labelsFeas)

		startMIPidx += 1

	print("Num. feasible MILP:", np.sum(Labels_feas),'/',ID_end - ID_start)

	if is_append_rand:
		print('Before appending:',ConFeatures_all.shape,VarFeatures_all.shape)
		kkk = ConFeatures_all.shape[0] // m
		np.random.seed(0)
		ConAug = np.tile(np.random.rand(m,1), (kkk,1))
		VarAug = np.tile(np.random.rand(n,1), (kkk,1))
		ConFeatures_all = np.concatenate((ConFeatures_all, ConAug),axis=1)
		VarFeatures_all = np.concatenate((VarFeatures_all, VarAug),axis=1)
		print('After appending:',ConFeatures_all.shape,VarFeatures_all.shape)
	
	if not os.path.exists(folder):
		os.mkdir(folder)
	np.savetxt(folder + '/ConFeatures_all.csv', ConFeatures_all, delimiter = ',', fmt = '%10.5f')
	np.savetxt(folder + '/EdgeFeatures_all.csv', EdgeFeatures_all, delimiter = ',', fmt = '%10.5f')
	np.savetxt(folder + '/EdgeIndices_all.csv', EdgeIndices_all, delimiter = ',', fmt = '%d')
	np.savetxt(folder + '/VarFeatures_all.csv', VarFeatures_all, delimiter = ',', fmt = '%10.5f')
	np.savetxt(folder + '/Labels_feas.csv', Labels_feas, delimiter = ',', fmt = '%10.5f')


def combineGraphsFeas(ID_start, ID_end, configs, folder):
	'''
	This function combines all feasible MILP instances in "folder".
	This function also makes labels for the optimal objective and optimal solution.
	'''
	m,n,MIP_key,is_append_rand = configs

	def my_stack(ori,aug):
		return (np.copy(aug) if ori is None else np.concatenate((ori,aug),axis=0) )

	ConFeatures_feas = None
	EdgeFeatures_feas = None
	EdgeIndices_feas = None
	VarFeatures_feas = None
	Labels_solu = None
	Labels_obj = None
	startMIPidx = 0

	path_list = []
	for k in range(ID_start, ID_end):
		MIP_path = MIP_key + '_' + str(k)
		if os.path.exists(MIP_path + '/Labels_solu.csv'):
			path_list.append(MIP_path)
	print("Num. feasible MILP:", len(path_list),'/',ID_end - ID_start)

	for MIP_path in path_list:
		varFeatures = read_csv(MIP_path + "/VarFeatures.csv", header=None).values
		conFeatures = read_csv(MIP_path + "/ConFeatures.csv", header=None).values
		edgeFeatures = read_csv(MIP_path + "/EdgeFeatures.csv", header=None).values
		edgeIndices = read_csv(MIP_path + "/EdgeIndices.csv", header=None).values
		labelsObj = read_csv(MIP_path + "/Labels_obj.csv", header=None).values
		labelsSolu = read_csv(MIP_path + "/Labels_solu.csv", header=None).values
		
		edgeIndices[:, 0] = edgeIndices[:, 0] + startMIPidx * m
		edgeIndices[:, 1] = edgeIndices[:, 1] + startMIPidx * n
		
		ConFeatures_feas = my_stack(ConFeatures_feas, conFeatures)
		VarFeatures_feas = my_stack(VarFeatures_feas, varFeatures)
		EdgeFeatures_feas = my_stack(EdgeFeatures_feas, edgeFeatures)
		EdgeIndices_feas = my_stack(EdgeIndices_feas, edgeIndices)
		Labels_solu = my_stack(Labels_solu, labelsSolu)
		Labels_obj = my_stack(Labels_obj, labelsObj)

		startMIPidx += 1

	if is_append_rand:
		print('Before appending:',ConFeatures_feas.shape,VarFeatures_feas.shape)
		kkk = ConFeatures_feas.shape[0] // m
		np.random.seed(0)
		ConAug = np.tile(np.random.rand(m,1), (kkk,1))
		VarAug = np.tile(np.random.rand(n,1), (kkk,1))
		ConFeatures_feas = np.concatenate((ConFeatures_feas, ConAug),axis=1)
		VarFeatures_feas = np.concatenate((VarFeatures_feas, VarAug),axis=1)
		print('After appending:',ConFeatures_feas.shape,VarFeatures_feas.shape)
		
	if not os.path.exists(folder):
		os.mkdir(folder)
	np.savetxt(folder + '/ConFeatures_feas.csv', ConFeatures_feas, delimiter = ',', fmt = '%10.5f')
	np.savetxt(folder + '/EdgeFeatures_feas.csv', EdgeFeatures_feas, delimiter = ',', fmt = '%10.5f')
	np.savetxt(folder + '/EdgeIndices_feas.csv', EdgeIndices_feas, delimiter = ',', fmt = '%d')
	np.savetxt(folder + '/VarFeatures_feas.csv', VarFeatures_feas, delimiter = ',', fmt = '%10.5f')
	np.savetxt(folder + '/Labels_solu.csv', Labels_solu, delimiter = ',', fmt = '%10.5f')
	np.savetxt(folder + '/Labels_obj.csv', Labels_obj, delimiter = ',', fmt = '%10.5f')	


## MAIN SCRIPT
if exp_env == 1:
	if not os.path.exists("./data-env1"):
		os.mkdir("./data-env1")
	generateMILPunfoldable(1000, (m,n,nnz,exp_env), "./data-MILPs/")
	combineGraphsAll(0,1000,(m,n,"./data-MILPs/MIP_env1_unfoldable",False), "./data-env1/unfoldable")
	combineGraphsFeas(0,1000,(m,n,"./data-MILPs/MIP_env1_unfoldable",False), "./data-env1/unfoldable")
	generateMILPfoldable(1000, (m,n,nnz,exp_env), "./data-MILPs/")
	combineGraphsAll(0,1000,(m,n,"./data-MILPs/MIP_env1_foldable",False), "./data-env1/foldable")
	combineGraphsFeas(0,1000,(m,n,"./data-MILPs/MIP_env1_foldable",False), "./data-env1/foldable")
	combineGraphsAll(0,1000,(m,n,"./data-MILPs/MIP_env1_foldable",True), "./data-env1/foldable-randFeat")
	combineGraphsFeas(0,1000,(m,n,"./data-MILPs/MIP_env1_foldable",True), "./data-env1/foldable-randFeat")
else:
	if not os.path.exists("./data-env2"):
		os.mkdir("./data-env2")
	generateMILPfoldable(4000, (m,n,nnz,exp_env), "./data-MILPs/")
	combineGraphsAll(0,2000,(m,n,"./data-MILPs/MIP_env2_foldable",True), "./data-env2/training")
	combineGraphsFeas(0,2000,(m,n,"./data-MILPs/MIP_env2_foldable",True), "./data-env2/training")
	combineGraphsAll(2000,4000,(m,n,"./data-MILPs/MIP_env2_foldable",True), "./data-env2/testing")
	combineGraphsFeas(2000,4000,(m,n,"./data-MILPs/MIP_env2_foldable",True), "./data-env2/testing")



