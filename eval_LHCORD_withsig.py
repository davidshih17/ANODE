import argparse
import copy
import math
import sys,os
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import flows as fnn

from eval_model import eval_model
from data_loader import load_dataset, load_data

parser = argparse.ArgumentParser(description='Evaluate ConditionalMAF on LHCORD dataset')

parser.add_argument('--innermodel', default='inner.par',
                    help='inner model')
parser.add_argument('--outermodel', default='outer.par',
                    help='outer model')
parser.add_argument('--datashift', default='0',
                    help='mj linear shift by mjj (default 0)')

results = parser.parse_args(sys.argv[1:])
print(results)

datashift=float(results.datashift)
innermodel=results.innermodel
outermodel=results.outermodel


# model parameters
flow='maf'
num_blocks = 15
num_hidden = 128
lr = 1e-4

print('MAF pars',num_blocks,num_hidden)

CUDA = False
device = torch.device("cuda:0" if CUDA else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}
kwargs = {}

features_sig=pd.read_hdf("/het/p4/dshih/jet_images-deep_learning/density_estimation/ConditionalMAF/events_anomalydetection_DelphesPythia8_v2_Wprime_features.h5")
features_bg=pd.read_hdf("/het/p4/dshih/jet_images-deep_learning/density_estimation/ConditionalMAF/events_anomalydetection_DelphesPythia8_v2_qcd_features.h5")

dataset_bg=load_data(features_bg)
dataset_bg=np.hstack((dataset_bg,np.zeros((len(dataset_bg),1))))

dataset_sig=load_data(features_sig)
dataset_sig=np.hstack((dataset_sig,np.ones((len(dataset_sig),1))))

dataset=np.concatenate((dataset_bg[:500000],dataset_sig[:500])).astype('float32') 
dataset_test=np.concatenate((dataset_bg[500000:],dataset_sig[500:])).astype('float32') 
dataset_test_actual=np.concatenate((dataset_bg[500000:],dataset_sig[500:1000])).astype('float32') 

dataset_shifted=np.copy(dataset)
dataset_shifted[:,1]=dataset_shifted[:,1]+datashift*(dataset_shifted[:,0]-3.5)
dataset_shifted[:,2]=dataset_shifted[:,2]+datashift*(dataset_shifted[:,0]-3.5)

dataset_test_shifted=np.copy(dataset_test)
dataset_test_shifted[:,1]=dataset_test_shifted[:,1]+datashift*(dataset_test_shifted[:,0]-3.5)
dataset_test_shifted[:,2]=dataset_test_shifted[:,2]+datashift*(dataset_test_shifted[:,0]-3.5)

dataset_test_actual_shifted=np.copy(dataset_test_actual)
dataset_test_actual_shifted[:,1]=dataset_test_actual_shifted[:,1]+datashift*(dataset_test_actual_shifted[:,0]-3.5)
dataset_test_actual_shifted[:,2]=dataset_test_actual_shifted[:,2]+datashift*(dataset_test_actual_shifted[:,0]-3.5)

traindict=load_dataset(dataset_shifted)
testdict=load_dataset(dataset_test_shifted)
testdict_actual=load_dataset(dataset_test_actual_shifted)

flow='maf'
num_blocks = 15
num_hidden = 128
lr = 1e-4
num_inputs = traindict['all_tensor'].shape[-1]
num_cond_inputs = traindict['all_labels'].shape[-1]  # labels are conditional data


CUDA = False
device = torch.device("cuda:0" if CUDA else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}
kwargs = {}

# Define model
act = 'relu'

modules = []
for _ in range(num_blocks):
    modules += [
        fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
        fnn.BatchNormFlow(num_inputs),
        fnn.Reverse(num_inputs)
    ]

outer_model = fnn.FlowSequential(*modules)
for module in outer_model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)
outer_model.to(device)
            

modules = []
for _ in range(num_blocks):
    modules += [
        fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
        fnn.BatchNormFlow(num_inputs),
        fnn.Reverse(num_inputs)
    ]

    
inner_model = fnn.FlowSequential(*modules)
for module in inner_model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)
inner_model.to(device)


print()

#latest_run="runs/Dec02_01-40-38_pascal.physics.rutgers.edumaf_LHCORD" #run with logit and standardization, using inner, larger network, SHIFTED DATASET

outer_model.load_state_dict(torch.load(outermodel, map_location='cpu'))
inner_model.load_state_dict(torch.load(innermodel, map_location='cpu'))

outer_model.eval()
inner_model.eval()


logPfull_actual=eval_model(inner_model,testdict_actual['inner_loader'],device,
           transform=(traindict['inner_max'],traindict['inner_min'],traindict['inner_mean2'],traindict['inner_std2']))
logPbg_actual=eval_model(outer_model,testdict_actual['inner_loader'],device,
       transform=(traindict['outer_max'],traindict['outer_min'],traindict['outer_mean2'],traindict['outer_std2']))    
inner_tensor_test_actual_rescaled=(testdict_actual['inner_tensor']-traindict['inner_min'])\
    /((traindict['inner_max']-traindict['inner_min']))
boxcut_test_actual=np.all(inner_tensor_test_actual_rescaled.numpy()>0.05,axis=1) & \
    np.all(inner_tensor_test_actual_rescaled.numpy()<.95,axis=1) & (logPbg_actual>-2) #& (pmphi2_out[:,1]>0)

np.save(outermodel+'_logPfull_actual',logPfull_actual)
np.save(outermodel+'_logPbg_actual',logPbg_actual)
np.save(outermodel+'_boxcut_actual',boxcut_test_actual)

logPfull=eval_model(inner_model,testdict['inner_loader'],device,
       transform=(traindict['inner_max'],traindict['inner_min'],traindict['inner_mean2'],traindict['inner_std2']))    
logPbg=eval_model(outer_model,testdict['inner_loader'],device,
       transform=(traindict['outer_max'],traindict['outer_min'],traindict['outer_mean2'],traindict['outer_std2']))    
inner_tensor_test_rescaled=(testdict['inner_tensor']-traindict['inner_min'])\
    /((traindict['inner_max']-traindict['inner_min']))
boxcut_test=np.all(inner_tensor_test_rescaled.numpy()>0.05,axis=1) & \
    np.all(inner_tensor_test_rescaled.numpy()<.95,axis=1) & (logPbg>-2)
np.save(outermodel+'_logPfull',logPfull)
np.save(outermodel+'_logPbg',logPbg)
np.save(outermodel+'_boxcut',boxcut_test)

    