## Read .npy

import numpy as np

#DAG
# Reading the CPDAG (Completed Partially Directed Acyclic Graph)
to_open = "/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_dp-dag/data_p100_e400_n1000_GP/CPDAG1.npy" #path to the .npy file
data=np.load(to_open)
print(data)

#DATA
# Reading the corresponding DAG
to_open2 = "/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_dp-dag/data_p100_e400_n1000_GP/DAG1.npy"
data2=np.load(to_open2)
print(data2)


## Visualise
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

# for the ground truth dags
to_open = "/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_salma/Poisson_p7_n1000/DAG1.npy"
data = np.load(to_open)
print(data)

n_nodes = len(data)

# create a directed graph object
gAdata= nx.DiGraph()
# add edges to our graph where there are 1s in our adj matrix
for i in range(0, n_nodes):
    for j in range(0, n_nodes):
        if data[i,j]==1:
            gAdata.add_edges_from([(i, j)])

# Draw and save the ground truth graph
plt.tight_layout()
nx.draw_networkx(gAdata, arrows=True)
plt.savefig("/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/visualisation/DAG_Poisson_true_p7_n1000.png", format="PNG")
plt.clf() # clear the figure for our next plot


# for the obtained results
# get out model's pred graph structure
data=results[1].probabilistic_dag.get_threshold_mask(threshold=0.14) # to convert our model's probabilistic predictions into a binary adjacency matrix
print(data)

n_nodes=len(data)

# Create and draw the predicted graph just like we did for the ground truth
gAdata= nx.DiGraph()
for i in range(0,n_nodes):
    for j in range(0,n_nodes):
        if data[i,j]==1:
            gAdata.add_edges_from([(i, j)])

plt.tight_layout()
nx.draw_networkx(gAdata, arrows=True)
plt.savefig("/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/visualisation/DAG_Poisson_predicted_p7_n1000_3.png", format="PNG")
plt.clf()