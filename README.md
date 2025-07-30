# Causal discovery with continuous optimization

This repository contains Python implementations of three chosen algorithms for causal structure learning from observational data. The methods leverage continuous optimization and deep learning frameworks to discover Directed Acyclic Graphs (DAGs).

The three models are:
* **NOTEARS**: A characterization of acyclicity as a smooth, differentiable function, allowing for gradient-based optimization.
* **DAG-GNN**: A Graph Neural Network (GNN) approach for structure learning.
* **VI-DP-DAG**: A variational inference framework for discovering probabilistic DAGs.

-------------------------------------

## Repo structure

This repo is organized into three main directories, one for each algorithm.
```
.
├── NOTEARS/
├── DAG-GNN/
├── VI-DP-DAG/
└── README.md
```
enjoy ~