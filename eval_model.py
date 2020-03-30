import torch
import numpy as np

def logit_transform(data,datamax,datamin):

    data2=(data-datamin)/(datamax-datamin)
    data3=torch.log((data2)/(1-data2))
    
    return data3


def eval_model(model,data_loader,device,transform=None):
    
    model.eval()
    allout=torch.empty(0)
    for batch_idx, data in enumerate(data_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None
            data = data[0]
        
        data = data.to(device)
        if(transform!=None):
            maxval,minval,meanval,stdval=transform
#            print(maxval,minval,meanval,stdval)
            data = logit_transform(data,maxval,minval)
            data = (data-meanval)/stdval
        output = model.log_probs(data, cond_data).flatten()
#        print(output)
        if(transform!=None):
            output=output+torch.sum(torch.log(2*(1+torch.cosh(data*stdval+meanval))/(stdval*(maxval-minval))),dim=1)
        allout = torch.cat((allout,output))
        
    allout=allout.cpu().detach().numpy()

    return allout
    
    
    
