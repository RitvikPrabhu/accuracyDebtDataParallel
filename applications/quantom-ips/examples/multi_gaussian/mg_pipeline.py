import torch

class MGPipeline:
    '''
    Simple pipeline to predict predictions of the format: [[mu1,sigma1],[mu2,sigma2]] to n datasets, provided the scaling matrix
    alpha = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    '''

    def __init__(self,n_events,alpha):
        self.n_events = []
        for a in alpha:
            self.n_events.append([int(j*n_events) for j in a[:3]])

    def create_single_gaussian(self,params,nevs):
        epsilon = torch.normal(mean=0.0,std=1.0,size=(nevs,),dtype=params[0].dtype,device=params[0].device)
        mean = torch.ones_like(epsilon)*params[0]
        sigma = torch.ones_like(epsilon)*params[1]
        return mean + sigma * epsilon
    
    def create_gaussian(self,params,nevs):
        return torch.vmap(lambda p: self.create_single_gaussian(p,nevs),in_dims=0,randomness="different")(params).transpose(0,1).flatten(0,1)

    def create_subdata(self,nevs,param_g1,param_g2):
        subset = []

        if nevs[0] > 0:
            s1 = self.create_gaussian(param_g1,nevs[0])
            subset.append(
                s1
            )
            
        if nevs[1] > 0:
            s2 = self.create_gaussian(param_g2,nevs[1])
            subset.append(
                s2
            )
            
        if nevs[2] > 0:
            norm = (param_g1[:,1]**2 + param_g2[:,1]**2)
            mu_prod = (param_g1[:,0]*(param_g2[:,1]**2) + param_g2[:,0]*(param_g1[:,1]**2)) / norm
            sigma_prod = torch.sqrt((param_g1[:,1]**2 * param_g2[:,1]**2) / norm)
            subset.append(
                self.create_gaussian((mu_prod,sigma_prod),nevs[2])
            )

        return torch.cat(subset)


    def forward(self,predictions,idx):
        param_g1 = predictions[0]
        param_g2 = predictions[1]

        if idx is not None:
           nevs = self.n_events[idx]
           return self.create_subdata(nevs, param_g1, param_g2)[:,None]
        else:
           datasets = []
           for nevs in self.n_events:
            w = self.create_subdata(nevs, param_g1, param_g2)[:,None]
            datasets.append(w)

           return torch.cat(datasets,1)