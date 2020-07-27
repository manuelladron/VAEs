import torch
from torch.distributions import bernoulli, normal
import numpy as np

def kl_loss(mu_z,logstd_z,reduce=True):
    '''
    Args:
        reduce: Whether to sum losses across features. Mean across samples is still done.
    '''

    # Assume standard normal prior    
    l1 = torch.exp(2*logstd_z)
    l2 = mu_z**2
    l3 = 2*logstd_z
    
    if reduce:
        return 0.5*(l1+l2-1-l3).sum(1).mean()
    else:
        return 0.5*(l1+l2-1-l3).mean(0)


def get_capacity_kl_loss(C):
    
    def func(mu_z,logstd_z,reduce=True):
        return kl_loss(mu_z,logstd_z,reduce) - C 
    
    return func

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
    # NOTE: Only pass 0 and 1s in x, otherwise we get absurd probabilities
    rv = bernoulli.Bernoulli(logits=logits.reshape(logits.shape[0],-1))
    return -rv.log_prob(x.reshape(logits.shape[0],-1)).sum(1).mean()


def log_prob_ratio_normal(z,mu_z_prior,logstd_z_prior,mu_z,logstd_z):
    
    return (-normal.Normal(mu_z_prior.flatten(start_dim=1),logstd_z_prior.exp().flatten(start_dim=1)).log_prob(z.flatten(start_dim=1)) + normal.Normal(mu_z.flatten(start_dim=1),logstd_z.exp().flatten(start_dim=1)).log_prob(z.flatten(start_dim=1))).sum(1).mean()

def makeLossLayered(loss,weights=None):
    
    def func(*args):
        if weights is None:
            return sum([  loss(*arg) for arg in zip(*args) ])
        else:
            return sum([ w*loss(*arg) for arg, w in zip(zip(*args),weights) ])

    return func
