# tensorflow=2.4
import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
from models import GCNPolicy

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--exp_env",    default='2',					choices = ['1','2'])
parser.add_argument("--data", 		help="number of testing data", 	default=1000)
parser.add_argument("--gpu", 		help="gpu index", 				default="0")
parser.add_argument("--data_path", 	default=None)
parser.add_argument("--model_path", default=None)
args = parser.parse_args()

## FUNCTION OF TRAINING PER EPOCH
def process(model, dataloader, type = 'fea', n_Vars_small = 20):

	c, ei, ev, v, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
	batched_states = (c, ei, ev, v, n_cs, n_vs, n_csm, n_vsm)  
	logits = model(batched_states, tf.convert_to_tensor(False)) 
	
	return_err = None
	
	if type == "fea":
		errs_fp = np.sum((logits.numpy() > 0.5) & (cand_scores.numpy() < 0.5))
		errs_fn = np.sum((logits.numpy() < 0.5) & (cand_scores.numpy() > 0.5))
		errs = errs_fp + errs_fn
		return_err = errs / cand_scores.shape[0]
	
	else:
		loss = tf.keras.metrics.mean_squared_error(cand_scores, logits)
		return_err = tf.reduce_mean(loss).numpy()

	return return_err

## SET-UP MODEL
model_path = args.model_path
embSize = int(model_path[:-4].split('-')[-1][1:])
type = model_path[:-4].split('-')[-3]

## SET-UP DATASET
datafolder = args.data_path
n_Samples_test = int(args.data)
n_Cons_small = 6 # Each MILP has 6 constraints
n_Vars_small = 20 # Each MILP has 20 variables
if "data-env1-unfoldable" in model_path:
	n_Eles_small = 60 # Each MILP has 60 nonzeros in matrix A
else:
	n_Eles_small = 12

## LOAD DATASET INTO MEMORY
if type == "fea":
	varFeatures = read_csv(datafolder + "/VarFeatures_all.csv", header=None).values[:n_Vars_small * n_Samples_test,:]
	conFeatures = read_csv(datafolder + "/ConFeatures_all.csv", header=None).values[:n_Cons_small * n_Samples_test,:]
	edgFeatures = read_csv(datafolder + "/EdgeFeatures_all.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	edgIndices = read_csv(datafolder + "/EdgeIndices_all.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	labels = read_csv(datafolder + "/Labels_feas.csv", header=None).values[:n_Samples_test,:]
if type == "obj":
	varFeatures = read_csv(datafolder + "/VarFeatures_feas.csv", header=None).values[:n_Vars_small * n_Samples_test,:]
	conFeatures = read_csv(datafolder + "/ConFeatures_feas.csv", header=None).values[:n_Cons_small * n_Samples_test,:]
	edgFeatures = read_csv(datafolder + "/EdgeFeatures_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	edgIndices = read_csv(datafolder + "/EdgeIndices_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	labels = read_csv(datafolder + "/Labels_obj.csv", header=None).values[:n_Samples_test,:]
if type == "sol":
	varFeatures = read_csv(datafolder + "/VarFeatures_feas.csv", header=None).values[:n_Vars_small * n_Samples_test,:]
	conFeatures = read_csv(datafolder + "/ConFeatures_feas.csv", header=None).values[:n_Cons_small * n_Samples_test,:]
	edgFeatures = read_csv(datafolder + "/EdgeFeatures_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	edgIndices = read_csv(datafolder + "/EdgeIndices_feas.csv", header=None).values[:n_Eles_small * n_Samples_test,:]
	labels = read_csv(datafolder + "/Labels_solu.csv", header=None).values[:n_Vars_small * n_Samples_test,:]

nConsF = conFeatures.shape[1]
nVarF = varFeatures.shape[1]
nEdgeF = edgFeatures.shape[1]
n_Cons = conFeatures.shape[0]
n_Vars = varFeatures.shape[0]

## SET-UP TENSORFLOW
gpu_index = int(args.gpu)
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

with tf.device("GPU:"+str(gpu_index)):

	### LOAD DATASET INTO GPU ###
	varFeatures = tf.constant(varFeatures, dtype=tf.float32)
	conFeatures = tf.constant(conFeatures, dtype=tf.float32)
	edgFeatures = tf.constant(edgFeatures, dtype=tf.float32)
	edgIndices = tf.constant(edgIndices, dtype=tf.int32)
	edgIndices = tf.transpose(edgIndices)
	labels = tf.constant(labels, dtype=tf.float32)
	data = (conFeatures, edgIndices, edgFeatures, varFeatures, n_Cons, n_Vars, n_Cons_small, n_Vars_small, labels)

	### LOAD MODEL ###
	if type == "sol":
		model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF, isGraphLevel = False)
	else:
		model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF)
	model.restore_state("./saved-models/" + model_path)

	### TEST MODEL ###
	err = process(model, data, type = type, n_Vars_small = n_Vars_small)
	model.summary()
	print(f"MODEL: {model_path}, DATA-SET: {datafolder}, NUM-DATA: {n_Samples_test}, EXP: {args.exp_env}, ERR: {err}")
	
	

