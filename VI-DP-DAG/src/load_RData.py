## Load the .RData files

import pyreadr
import numpy as np
import pandas as pd



## ZIP

#DAG
resultDAG = pyreadr.read_r('/Users/salma/Desktop/Cambridge/zipbn-main/Training/DAG_ZIP.RData')

DAG_zip_ordered_dict = resultDAG
DAG_zip = np.array(list(DAG_zip_ordered_dict.values()))
DAG_zip=DAG_zip[0]
print(DAG_zip)
np.save('/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_salma/Test_Bottolo_ER/DAG1.npy', DAG_zip)

#DATA

resultdata = pyreadr.read_r('/Users/salma/Desktop/Cambridge/zipbn-main/Training/data_ZIP.RData')

data_zip_ordered_dict = resultdata
data_zip = np.array(list(data_zip_ordered_dict.values()))
data_zip=data_zip[0]
print(data_zip)
np.save('/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_salma/Test_Bottolo_ER/data1.npy', data_zip)


## Poisson

#DAG
resultDAG = pyreadr.read_r('/Users/salma/Desktop/Cambridge/zipbn-main/Training/DAG_poisson.RData')

DAG_poisson_ordered_dict = resultDAG
DAG_poisson = np.array(list(DAG_poisson_ordered_dict.values()))
DAG_poisson=DAG_poisson[0]
print(DAG_poisson)
np.save('/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_salma/Poisson_p7_n1000/DAG1.npy', DAG_poisson)

#DATA

resultdata = pyreadr.read_r('/Users/salma/Desktop/Cambridge/zipbn-main/Training/data_poisson.RData')

data_poisson_ordered_dict = resultdata
data_poisson = np.array(list(data_poisson_ordered_dict.values()))
data_poisson=data_poisson[0]
print(data_poisson)
np.save('/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_salma/Poisson_p7_n1000/data1.npy', data_poisson)



## NB


#DAG
resultDAG = pyreadr.read_r('/Users/salma/Desktop/Cambridge/zipbn-main/Training/DAG_NB.RData')

DAG_NB_ordered_dict = resultDAG
DAG_NB = np.array(list(DAG_NB_ordered_dict.values()))
DAG_NB=DAG_NB[0]
print(DAG_NB)
np.save('/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_salma/NB_p10_r1/DAG1.npy', DAG_NB)

#DATA

resultdata = pyreadr.read_r('/Users/salma/Desktop/Cambridge/zipbn-main/Training/data_NB.RData')

data_NB_ordered_dict = resultdata
data_NB = np.array(list(data_NB_ordered_dict.values()))
data_NB=data_NB[0]
print(data_NB)
np.save('/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_salma/NB_p10_r1/data1.npy', data_NB)


## ZINB

#DAG
resultDAG = pyreadr.read_r('/Users/salma/Desktop/Cambridge/zipbn-main/Training/DAG_ZINB.RData')

DAG_ZINB_ordered_dict = resultDAG
DAG_ZINB = np.array(list(DAG_ZINB_ordered_dict.values()))
DAG_ZINB=DAG_ZINB[0]
print(DAG_ZINB)
np.save('/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_salma/ZINB_p50_r2/DAG1.npy', DAG_ZINB)

#DATA

resultdata = pyreadr.read_r('/Users/salma/Desktop/Cambridge/zipbn-main/Training/data_ZINB.RData')

data_ZINB_ordered_dict = resultdata
data_ZINB = np.array(list(data_ZINB_ordered_dict.values()))
data_ZINB=data_ZINB[0]
print(data_ZINB)
np.save('/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_salma/ZINB_p50_r2/data1.npy', data_ZINB)



## Normal


#DAG
resultDAG = pyreadr.read_r('/Users/salma/Desktop/Cambridge/zipbn-main/Training/DAG_gauss.RData')

DAG_gauss_ordered_dict = resultDAG
DAG_gauss = np.array(list(DAG_gauss_ordered_dict.values()))
DAG_gauss=DAG_gauss[0]
print(DAG_gauss)
np.save('/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_salma/Gauss_new_p50/DAG1.npy', DAG_gauss)

#DATA

resultdata = pyreadr.read_r('/Users/salma/Desktop/Cambridge/zipbn-main/Training/data_gauss.RData')

data_gauss_ordered_dict = resultdata
data_gauss = np.array(list(data_gauss_ordered_dict.values()))
data_gauss=data_gauss[0]
print(data_gauss)
np.save('/Users/salma/Desktop/Cambridge/Differentiable-DAG-Sampling-master/datasets_salma/Gauss_new_p50/data1.npy', data_gauss)
