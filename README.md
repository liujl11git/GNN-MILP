# On Representing Mixed-Integer Linear Programs by Graph Neural Networks


This repository is an implementation of the paper entitled "On Representing Mixed-Integer Linear Programs by Graph Neural Networks." (ICLR 2023) The paper can be found [here](https://openreview.net/forum?id=4gc3MGZra1d). Our codes are modified from [this repo](https://github.com/ds4dm/learn2branch).

## Introduction

While Mixed-integer linear programming (MILP) is NP-hard in general, practical MILP has received roughly 100--fold speedup in the past twenty years. Still, many classes of MILPs quickly become unsolvable as their sizes increase, motivating researchers to seek new acceleration techniques for MILPs. With deep learning, they have obtained strong empirical results, and many results were obtained by applying graph neural networks (GNNs) to making decisions in various stages of MILP solution processes. This work discovers a fundamental limitation: there exist feasible and infeasible MILPs that all GNNs will, however, treat equally, indicating GNN's lacking power to express general MILPs. Then, we show that, by restricting the MILPs to unfoldable ones or by adding random features, there exist GNNs that can reliably predict MILP feasibility, optimal objective values, and optimal solutions up to prescribed precision.  We conducted small-scale numerical experiments to validate our theoretical findings.

## A quick start guide

Step 1: Generating enough data
```
python 1_generate_data.py --exp_env 2 
```
Step 2: Training and testing a GNN for the feasibility of MILP.
```
python 2_training.py --type fea --data 1000 --data_path data-env2/training --embSize 8
python 3_testing.py --data_path data-env2/testing --model_path data-env2-training-fea-d1000-s8.pkl
```
Step 3: Training and testing a GNN for the objective of MILP.
```
python 2_training.py --type obj --data 1000 --data_path data-env2/training --embSize 8
python 3_testing.py --data_path data-env2/testing --model_path data-env2-training-obj-d1000-s8.pkl
```
Step 4: Training and testing a GNN for the solution of MILP.
```
python 2_training.py --type sol --data 1000 --data_path data-env2/training --embSize 8
python 3_testing.py --data_path data-env2/testing --model_path data-env2-training-sol-d1000-s8.pkl
```

## Reproducing all results

To reproduce all the results, please follow the commands in "cmds.txt"

Our environment: NVIDIA Tesla V100, CUDA 10.1, tensorflow 2.4.1, PySCIPOpt 4.2.0.

## Related repo

On Representing Linear Programs by Graph Neural Networks:

https://github.com/liujl11git/GNN-LP

## Citing our work

If you find our code helpful in your resarch or work, please cite our paper.
```
@inproceedings{
chen2023gnn-mip,
title={On Representing Mixed-Integer Linear Programs by Graph Neural Networks},
author={Ziang Chen and Jialin Liu and Xinshang Wang and Jianfeng Lu and Wotao Yin},
booktitle={International Conference on Learning Representations},
year={2023}
}
```

