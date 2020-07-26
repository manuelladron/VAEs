import torch
from torch.distributions import bernoulli, normal
import numpy as np

def kl_loss(mu_z,logstd_z):
    # Assume standard normal prior    
    l1 = torch.exp(2*logstd_z)
    l2 = mu_z**2
    l3 = 2*logstd_z

    return 0.5*(l1+l2-1-l3).sum(1).mean()

def recon_loss(x,x_recon,logstd_noise):
    
    # Removing constant term for better resolution of the optimization
    #l1 = torch.log(torch.tensor(2*np.pi,device=device))
    l2 = 2*logstd_noise
    l3 = ((x-x_recon)**2/torch.exp(2*logstd_noise))

    return 0.5*(l2+l3).reshape(x.shape[0],-1).sum(1).mean()

def elbo_loss(x,x_recon,logstd_noise,mu_z,logstd_z,device):
    
    kl = kl_loss(mu_z,logstd_z)
    recon = recon_loss(x,x_recon,logstd_noise,device)

    return kl + recon, kl, recon

def recon_loss_bernoulli(x,logits,*args,**kwargs):
    # *args to take in extra variables to fit in the existing trainer setup
    rv = bernoulli.Bernoulli(logits=logits)
    return -rv.log_prob(x).sum(1).mean()


def log_prob_ratio_normal(z,mu_z_prior,logstd_z_prior,mu_z,logstd_z):
    
    return (-normal.Normal(mu_z_prior.flatten(start_dim=1),logstd_z_prior.exp().flatten(start_dim=1)).log_prob(z.flatten(start_dim=1)) + normal.Normal(mu_z.flatten(start_dim=1),logstd_z.exp().flatten(start_dim=1)).log_prob(z.flatten(start_dim=1))).sum(1).mean()

def makeLossLayered(loss):
    
    def func(*args):
       return sum([  loss(*arg) for arg in zip(*args) ])

    return func
