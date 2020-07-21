import torch
import numpy as np

def kl_loss(mu_z,logstd_z):
    
    l1 = torch.exp(2*logstd_z)
    l2 = mu_z**2
    l3 = 2*logstd_z

    return 0.5*(l1+l2-1-l3).sum(1).mean()

def recon_loss(x,x_recon,logstd_noise,device):
    
    # Removing constant term for better resolution of the optimization
    #l1 = torch.log(torch.tensor(2*np.pi,device=device))
    l2 = 2*logstd_noise
    l3 = ((x-x_recon)**2/torch.exp(2*logstd_noise))

    return 0.5*(l2+l3).reshape(x.shape[0],-1).sum(1).mean()

def elbo_loss(x,x_recon,logstd_noise,mu_z,logstd_z,device):
    
    kl = kl_loss(mu_z,logstd_z)
    recon = recon_loss(x,x_recon,logstd_noise,device)

    return kl + recon, kl, recon
