# tensorflow=2.4
import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
import os
from models import GCNPolicy

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--exp_env", default='2', choices = ['1','2'])
parser.add_argument("--data", help="number of testing data", default=1000)
parser.add_argument("--set", help="which set you want to test on?", default="train", choices = ['test','train'])
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--model_key", default=None)
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

## SET-UP
temp = args.model_key.split('-')
type = temp[-1]
if args.set == "test" and args.exp_env == "2":
	temp[2] = "testing"
datafolder = temp[0]+'-'+temp[1]+'/'+'-'.join(s for s in temp[2:-1])
n_Samples_test = int(args.data)
n_Cons_small = 6 # Each MILP has 6 constraints
n_Vars_small = 20 # Each MILP has 20 variables
n_Eles_small = 60 if "data-env1-unfoldable" in args.model_key else 12
exp_list = []
for model_name in os.listdir("./saved-models"):
	if args.model_key not in model_name:
		continue
	model_path = "./saved-models/" + model_name
	embSize = int(model_name[:-4].split('-')[-1][1:])
	n_Samples = int(model_name.split('-')[-2][1:]) if args.set == "train" else n_Samples_test
	exp_list.append((model_path, embSize, n_Samples))

## LOAD DATASET INTO MEMORY
if type == "fea":
	varFeatures_np = read_csv(datafolder + "/VarFeatures_all.csv", header=None).values
	conFeatures_np = read_csv(datafolder + "/ConFeatures_all.csv", header=None).values
	edgFeatures_np = read_csv(datafolder + "/EdgeFeatures_all.csv", header=None).values
	edgIndices_np = read_csv(datafolder + "/EdgeIndices_all.csv", header=None).values
	labels_np = read_csv(datafolder + "/Labels_feas.csv", header=None).values
if type == "obj":
	varFeatures_np = read_csv(datafolder + "/VarFeatures_feas.csv", header=None).values
	conFeatures_np = read_csv(datafolder + "/ConFeatures_feas.csv", header=None).values
	edgFeatures_np = read_csv(datafolder + "/EdgeFeatures_feas.csv", header=None).values
	edgIndices_np = read_csv(datafolder + "/EdgeIndices_feas.csv", header=None).values
	labels_np = read_csv(datafolder + "/Labels_obj.csv", header=None).values
if type == "sol":
	varFeatures_np = read_csv(datafolder + "/VarFeatures_feas.csv", header=None).values
	conFeatures_np = read_csv(datafolder + "/ConFeatures_feas.csv", header=None).values
	edgFeatures_np = read_csv(datafolder + "/EdgeFeatures_feas.csv", header=None).values
	edgIndices_np = read_csv(datafolder + "/EdgeIndices_feas.csv", header=None).values
	labels_np = read_csv(datafolder + "/Labels_solu.csv", header=None).values

## SET-UP TENSORFLOW
gpu_index = int(args.gpu)
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

with tf.device("GPU:"+str(gpu_index)):

	for model_path, embSize, n_Samples in exp_list:

		### LOAD DATASET INTO GPU ###
		varFeatures = tf.constant(varFeatures_np[:n_Vars_small * n_Samples,:], dtype=tf.float32)
		conFeatures = tf.constant(conFeatures_np[:n_Cons_small * n_Samples,:], dtype=tf.float32)
		edgFeatures = tf.constant(edgFeatures_np[:n_Eles_small * n_Samples,:], dtype=tf.float32)
		edgIndices = tf.constant(edgIndices_np[:n_Eles_small * n_Samples,:], dtype=tf.int32)
		edgIndices = tf.transpose(edgIndices)
		if type == "sol":
			labels = tf.constant(labels_np[:n_Vars_small * n_Samples,:], dtype=tf.float32)
		else:
			labels = tf.constant(labels_np[:n_Samples,:], dtype=tf.float32)
		nConsF = conFeatures.shape[1]
		nVarF = varFeatures.shape[1]
		nEdgeF = edgFeatures.shape[1]
		n_Cons = conFeatures.shape[0]
		n_Vars = varFeatures.shape[0]
		data = (conFeatures, edgIndices, edgFeatures, varFeatures, n_Cons, n_Vars, n_Cons_small, n_Vars_small, labels)

		### LOAD MODEL ###
		if type == "sol":
			model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF, isGraphLevel = False)
		else:
			model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF)
		model.restore_state(model_path)

		### TEST MODEL ###
		err = process(model, data, type = type, n_Vars_small = n_Vars_small)
		print(f"MODEL: {model_path}, DATA-SET: {datafolder}, NUM-DATA: {n_Samples}, EXP: {args.exp_env}, ERR: {err}")
	
	

