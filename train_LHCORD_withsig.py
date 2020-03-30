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

parser = argparse.ArgumentParser(description='Train ConditionalMAF on LHCORD dataset')

parser.add_argument('--epochs', default='10',
                    help='epochs (default 10)')
parser.add_argument('--datashift', default='0',
                    help='mj linear shift by mjj (default 0)')
parser.add_argument('--label', default='0',
                    help='label for this run')
parser.add_argument('--minmass', default='3.3',
                    help='min mass for SR')
parser.add_argument('--maxmass', default='3.7',
                    help='max mass for SR')
parser.add_argument('--noshuffle', action='store_true',
                    help='do not shuffle randomly sample from bg set')
parser.add_argument('--transform', action='store_true',
                    help='logit transform the inputs')
parser.add_argument('--standardize', action='store_true',
                    help='standardize inputs')

results = parser.parse_args(sys.argv[1:])
print(results)

epochs=int(results.epochs)
datashift=float(results.datashift)
label=results.label
minmass=float(results.minmass)
maxmass=float(results.maxmass)
noshuffle=results.noshuffle
transform=results.transform
standardize=results.standardize

# model parameters
flow='maf'
batch_size = 256
test_batch_size = 10*batch_size
num_blocks = 15
num_hidden = 128
lr = 1e-4

print('MAF pars',num_blocks,num_hidden)

CUDA = False
device = torch.device("cuda:0" if CUDA else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}
kwargs = {}

features_sig=pd.read_hdf("LHCORD_sig_features.h5")
features_bg=pd.read_hdf("LHCORD_qcd_features.h5")

mj1mj2_bg=np.array(features_bg[['mj1','mj2']])
mj1mj2_sig=np.array(features_sig[['mj1','mj2']])


tau21_bg=np.array(features_bg[['tau2j1','tau2j2']])/(1e-5+np.array(features_bg[['tau1j1','tau1j2']]))
tau21_sig=np.array(features_sig[['tau2j1','tau2j2']])/(1e-5+np.array(features_sig[['tau1j1','tau1j2']]))


mjmin_bg=mj1mj2_bg[range(len(mj1mj2_bg)),np.argmin(mj1mj2_bg,axis=1)]
mjmax_bg=mj1mj2_bg[range(len(mj1mj2_bg)),np.argmax(mj1mj2_bg,axis=1)]
tau21min_bg=tau21_bg[range(len(mj1mj2_bg)),np.argmin(mj1mj2_bg,axis=1)]
tau21max_bg=tau21_bg[range(len(mj1mj2_bg)),np.argmax(mj1mj2_bg,axis=1)]

mjmin_sig=mj1mj2_sig[range(len(mj1mj2_sig)),np.argmin(mj1mj2_sig,axis=1)]
mjmax_sig=mj1mj2_sig[range(len(mj1mj2_sig)),np.argmax(mj1mj2_sig,axis=1)]
tau21min_sig=tau21_sig[range(len(mj1mj2_sig)),np.argmin(mj1mj2_sig,axis=1)]
tau21max_sig=tau21_sig[range(len(mj1mj2_sig)),np.argmax(mj1mj2_sig,axis=1)]


pjj_sig=(np.array(features_sig[['pxj1','pyj1','pzj1']])+np.array(features_sig[['pxj2','pyj2','pzj2']]))
Ejj_sig=np.sqrt(np.sum(np.array(features_sig[['pxj1','pyj1','pzj1','mj1']])**2,axis=1))\
    +np.sqrt(np.sum(np.array(features_sig[['pxj2','pyj2','pzj2','mj2']])**2,axis=1))
mjj_sig=np.sqrt(Ejj_sig**2-np.sum(pjj_sig**2,axis=1))

pjj_bg=(np.array(features_bg[['pxj1','pyj1','pzj1']])+np.array(features_bg[['pxj2','pyj2','pzj2']]))
Ejj_bg=np.sqrt(np.sum(np.array(features_bg[['pxj1','pyj1','pzj1','mj1']])**2,axis=1))\
    +np.sqrt(np.sum(np.array(features_bg[['pxj2','pyj2','pzj2','mj2']])**2,axis=1))
mjj_bg=np.sqrt(Ejj_bg**2-np.sum(pjj_bg**2,axis=1))

dataset_bg=np.dstack((mjj_bg/1000,mjmin_bg/1000,(mjmax_bg-mjmin_bg)/1000,tau21min_bg,tau21max_bg,np.zeros(len(mjj_bg))))[0]
dataset_sig=np.dstack((mjj_sig/1000,mjmin_sig/1000,(mjmax_sig-mjmin_sig)/1000,tau21min_sig,tau21max_sig,np.ones(len(mjj_sig))))[0]

indices=np.array(range(len(dataset_bg))).astype('int')

if(not noshuffle):
    print('Shuffling dataset')
    np.random.shuffle(indices)

indices=indices[:500000]
np.savetxt("indices_"+label+".csv",indices,delimiter=",",fmt='%i')

dataset=np.concatenate((dataset_bg[indices],dataset_sig[:500])).astype('float32')

dataset_shifted=np.copy(dataset)
dataset_shifted[:,1]=dataset_shifted[:,1]+datashift*(dataset_shifted[:,0]-3.5)
dataset_shifted[:,2]=dataset_shifted[:,2]+datashift*(dataset_shifted[:,0]-3.5)

innermask=(dataset_shifted[:,0]>minmass) & (dataset_shifted[:,0]<maxmass)
outermask=~innermask
sigmask=dataset[:,-1]==1
bgmask=dataset[:,-1]==0

Stry=len(dataset_shifted[innermask & sigmask])
Btry=len(dataset_shifted[innermask & bgmask])
print([Stry,Btry,Stry/Btry,Stry/np.sqrt(Btry)])

def logit_transform(data,labels,datamax,datamin):
#    datamax=torch.max(data,dim=0).values
#    datamin=torch.min(data,dim=0).values

    data2=(data-datamin)/(datamax-datamin)
    
    mask=(data2[:,0]>0) & (data2[:,0]<1) &\
               (data2[:,1]>0) & (data2[:,1]<1) &\
               (data2[:,2]>0) & (data2[:,2]<1) & \
                (data2[:,3]>0) & (data2[:,3]<1)
                
    data3=data2[mask]

    data4=torch.log((data3)/(1-data3))
    return data4,labels[mask],mask

if(transform):
    print('Applying logit transform')

if(standardize):
    print('Mean/std standardization')

print('Mode: inner vs outer')

all_labels = torch.from_numpy(dataset_shifted[:,0:1])
all_tensor = torch.from_numpy(dataset_shifted[:,1:-1])
all_max=torch.max(all_tensor,dim=0).values
all_min=torch.min(all_tensor,dim=0).values
if(transform):
    all_tensor2,all_labels2,_=logit_transform(all_tensor,all_labels,all_max,all_min)
else:
    all_tensor2,all_labels2=all_tensor,all_labels
#if(standardize):
#    all_tensor2=(all_tensor2-torch.mean(all_tensor2,dim=0))/torch.std(all_tensor2,dim=0)  
#all_dataset = torch.utils.data.TensorDataset(all_tensor2, all_labels2)
##all_dataset = torch.utils.data.TensorDataset(all_tensor, all_labels)


outer_labels = torch.from_numpy(dataset_shifted[outermask][:,0:1])
outer_tensor = torch.from_numpy(dataset_shifted[outermask][:,1:-1])
outer_max=torch.max(outer_tensor,dim=0).values
outer_min=torch.min(outer_tensor,dim=0).values
if(transform):
    outer_tensor2,outer_labels2,_=logit_transform(outer_tensor,outer_labels,outer_max,outer_min) # use the same lower and upper #    outer_tensor2,outer_labels2,_=logit_transform(outer_tensor,outer_labels,all_max,all_min) # use the same lower and upper limits for the logit transform
else:
    outer_tensor2,outer_labels2=outer_tensor,outer_labels
if(standardize):
    outer_tensor2=(outer_tensor2-torch.mean(outer_tensor2,dim=0))/torch.std(outer_tensor2,dim=0)
#    outer_tensor2=(outer_tensor2-torch.mean(all_tensor2,dim=0))/torch.std(all_tensor2,dim=0)  
outer_dataset = torch.utils.data.TensorDataset(outer_tensor2, outer_labels2)
#outer_dataset = torch.utils.data.TensorDataset(outer_tensor, outer_labels)


inner_labels = torch.from_numpy(dataset_shifted[innermask][:,0:1])
inner_tensor = torch.from_numpy(dataset_shifted[innermask][:,1:-1])
inner_max=torch.max(inner_tensor,dim=0).values
inner_min=torch.min(inner_tensor,dim=0).values
if(transform):
    inner_tensor2,inner_labels2,_=logit_transform(inner_tensor,inner_labels,inner_max,inner_min) # use the same lower and upper limits for the logit transform
else:
    inner_tensor2,inner_labels2=inner_tensor,inner_labels
if(standardize):
    inner_tensor2=(inner_tensor2-torch.mean(inner_tensor2,dim=0))/torch.std(inner_tensor2,dim=0)
#    inner_tensor2=(inner_tensor2-torch.mean(all_tensor2,dim=0))/torch.std(all_tensor2,dim=0)  
inner_dataset = torch.utils.data.TensorDataset(inner_tensor2, inner_labels2)
#inner_dataset = torch.utils.data.TensorDataset(inner_tensor, inner_labels)



num_inputs = outer_tensor.shape[-1]
num_cond_inputs = outer_labels.shape[-1]  # labels are conditional data

# DataLoaders

#all_loader = torch.utils.data.DataLoader(
#    all_dataset, batch_size=batch_size, shuffle=True)#, **kwargs)
outer_loader = torch.utils.data.DataLoader(
    outer_dataset, batch_size=batch_size, shuffle=True)#, **kwargs)
inner_loader = torch.utils.data.DataLoader(
    inner_dataset, batch_size=batch_size, shuffle=True)#, **kwargs)

global_step = 0


def train(epoch):
    global global_step
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)

        optimizer.zero_grad()
        # print(data, cond_data)
        loss = -model.log_probs(data, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))

        global_step += 1

    pbar.close()

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    # if args.cond:
    with torch.no_grad():
        model(train_loader.dataset.tensors[0].to(data.device),
            train_loader.dataset.tensors[1].to(data.device).float())

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1

#writer = SummaryWriter(comment="_"+flow + "_LHCORD_"+label)

#for model_name, train_loader in zip(["all", "outer"], [all_loader, outer_loader]):
#for model_name, train_loader in zip(["inner"], [inner_loader]):
for model_name, train_loader in zip(["outer"], [outer_loader]):
#for model_name, train_loader in zip(["inner", "outer"], [inner_loader, outer_loader]):
    # Define model
    act = 'relu'

    modules = []
    for _ in range(num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]

    model = fnn.FlowSequential(*modules)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    
    # train
    for epoch in range(epochs):
        print('\nEpoch: {}'.format(epoch))
        train(epoch)
        torch.save(model.state_dict(),model_name+"_"+label+"_epoch_"+str(epoch)+"_Model.par")

