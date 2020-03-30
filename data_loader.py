import torch
import numpy as np

def logit_transform(data,datamax,datamin):

    data2=(data-datamin)/(datamax-datamin)
    data3=torch.log((data2)/(1-data2))
    
    return data3

def create_dataset(data):
    sigorbg=torch.from_numpy(data[:,-1])
    labels = torch.from_numpy(data[:,0:1])
    tensor = torch.from_numpy(data[:,1:-1])
    dataset = torch.utils.data.TensorDataset(tensor, labels)

    return sigorbg,labels,tensor,dataset
    
    

def load_dataset(data):
    datadict={}
    
    innermask=(data[:,0]>3.3) & (data[:,0]<3.7)
    outermask=~innermask
    sigmask=data[:,-1]==1
    bgmask=data[:,-1]==0
    
    datadict['innermask']=innermask
    datadict['outermask']=outermask
    datadict['sigmask']=sigmask
    datadict['bgmask']=bgmask

    Stry=len(data[innermask & sigmask])
    Btry=len(data[innermask & bgmask])
    print([Stry,Btry,Stry/Btry,Stry/np.sqrt(Btry)])

    outer_sigorbg,outer_labels,outer_tensor,outer_dataset=create_dataset(data[outermask])
    datadict['outer_sigorbg']=outer_sigorbg
    datadict['outer_labels']=outer_labels
    datadict['outer_tensor']=outer_tensor
    datadict['outer_dataset']=outer_dataset

    inner_sigorbg,inner_labels,inner_tensor,inner_dataset=create_dataset(data[innermask])
    datadict['inner_sigorbg']=inner_sigorbg
    datadict['inner_labels']=inner_labels
    datadict['inner_tensor']=inner_tensor
    datadict['inner_dataset']=inner_dataset

    all_sigorbg,all_labels,all_tensor,all_dataset=create_dataset(data)
    datadict['all_sigorbg']=all_sigorbg
    datadict['all_labels']=all_labels
    datadict['all_tensor']=all_tensor
    datadict['all_dataset']=all_dataset

#    signal_labels = torch.from_numpy(dataset[innermask & sigmask][:,0:1])
#    signal_tensor = torch.from_numpy(dataset[innermask & sigmask][:,1:-1])
#    signal_dataset = torch.utils.data.TensorDataset(signal_tensor, signal_labels)



    batch_size = 256
    test_batch_size = 1000*batch_size

    # DataLoaders

    datadict['all_loader'] = torch.utils.data.DataLoader(
    all_dataset, batch_size=test_batch_size, shuffle=False)#, **kwargs)
    datadict['outer_loader'] = torch.utils.data.DataLoader(
    outer_dataset, batch_size=test_batch_size, shuffle=False)#, **kwargs)
    datadict['inner_loader'] = torch.utils.data.DataLoader(
    inner_dataset, batch_size=test_batch_size, shuffle=False)#, **kwargs)
#    signal_loader = torch.utils.data.DataLoader(
#    signal_dataset, batch_size=test_batch_size, shuffle=False)#, **kwargs)


    datadict['inner_max']=torch.max(inner_tensor,dim=0).values
    datadict['inner_min']=torch.min(inner_tensor,dim=0).values

    datadict['outer_max']=torch.max(outer_tensor,dim=0).values
    datadict['outer_min']=torch.min(outer_tensor,dim=0).values

    datadict['all_max']=torch.max(all_tensor,dim=0).values
    datadict['all_min']=torch.min(all_tensor,dim=0).values
    
    outer_tensor2=logit_transform(outer_tensor,datadict['outer_max'],datadict['outer_min'])
    outer_tensor2[outer_tensor2!=outer_tensor2]=0
    outer_tensor2[outer_tensor2==float('inf')]=0
    outer_tensor2[outer_tensor2==float('-inf')]=0    
    outer_mean2=torch.mean(outer_tensor2,dim=0)
    outer_std2=torch.std(outer_tensor2,dim=0)
    datadict['outer_mean2']=outer_mean2
    datadict['outer_std2']=outer_std2

    inner_tensor2=logit_transform(inner_tensor,datadict['inner_max'],datadict['inner_min'])
    inner_tensor2[inner_tensor2!=inner_tensor2]=0
    inner_tensor2[inner_tensor2==float('inf')]=0
    inner_tensor2[inner_tensor2==float('-inf')]=0    
    inner_mean2=torch.mean(inner_tensor2,dim=0)
    inner_std2=torch.std(inner_tensor2,dim=0)
    datadict['inner_mean2']=inner_mean2
    datadict['inner_std2']=inner_std2

    all_tensor2=logit_transform(all_tensor,datadict['all_max'],datadict['all_min'])
    all_tensor2[all_tensor2!=all_tensor2]=0
    all_tensor2[all_tensor2==float('inf')]=0
    all_tensor2[all_tensor2==float('-inf')]=0    
    all_mean2=torch.mean(all_tensor2,dim=0)
    all_std2=torch.std(all_tensor2,dim=0)
    datadict['all_mean2']=all_mean2
    datadict['all_std2']=all_std2

    
    return datadict

def load_data(features):

    pjj=(np.array(features[['pxj1','pyj1','pzj1']])+np.array(features[['pxj2','pyj2','pzj2']]))
    Ejj=np.sqrt(np.sum(np.array(features[['pxj1','pyj1','pzj1','mj1']])**2,axis=1))\
        +np.sqrt(np.sum(np.array(features[['pxj2','pyj2','pzj2','mj2']])**2,axis=1))
    mjj=np.sqrt(Ejj**2-np.sum(pjj**2,axis=1))

    mj1mj2=np.array(features[['mj1','mj2']])
    argminlist=np.argmin(mj1mj2,axis=1)
    argmaxlist=np.argmax(mj1mj2,axis=1)

    tau21=np.array(features[['tau2j1','tau2j2']])/(1e-5+np.array(features[['tau1j1','tau1j2']]))
    
    mjmin=mj1mj2[range(len(mj1mj2)),argminlist]
    mjmax=mj1mj2[range(len(mj1mj2)),argmaxlist]
    
    tau21min=tau21[range(len(mj1mj2)),argminlist]
    tau21max=tau21[range(len(mj1mj2)),argmaxlist]
        
    dataset=np.dstack((mjj/1000,
                           mjmin/1000,
                           (mjmax-mjmin)/1000,
                           tau21min,
                           tau21max))[0]

    return dataset




