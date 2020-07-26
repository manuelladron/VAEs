import torch
import torch.nn as nn
import copy
from .helper import activations

class Encoder(nn.Module):
    """ Models the mean and log standard deviation for q(z|x) which is assumed
    to be a gaussian with diagnol covariance matrix.
    """

    def __init__(self,input_dim,hiddens,latent_dim,shared=True):
        """
        Args:
            input_dim (int)
            hiddens (list): describes the configuration for hidden layers. Each
                element is an
                {
                    'dim':(int),
                    'act':(activation), # nn.ReLU()
                    'bn':(bool), # batch-norm after this layer or not
                }
            latent_dim (int)
            shared (bool): Whether sigma and mu are constructed from the same
                network. This may help in reducing no.of parameters in the net.
                
        """
        super(Encoder, self).__init__()
        
        hiddens.append({'dim': 2*latent_dim if shared else latent_dim,\
            'act':'identity','bn':False})
        
        prev_dims = [input_dim] + [ h['dim'] for h in hiddens[:-1] ]

        layers = [ l for indim, d in zip(prev_dims,hiddens) for l in \
            ((nn.Linear(indim,d['dim']),nn.BatchNorm1d(d['dim']),\
            activations[d.get('act','relu')]()) if d.get('bn',False) else \
            (nn.Linear(indim,d['dim']), activations[d.get('act','relu')]()))]
        
        self.net = nn.Sequential(*layers)
        self.shared = shared

        if not shared:
            self.net2 = copy.deepcopy(self.net)

    def forward(self,x):
        """
        
        Args:
            x (tensor): #batch x #features

        Return:
            mu_z (tensor): #batch x latent_dim This is the mean
            logstd_z (tensor): #batch x latent_dim This is log std
        """
        if not self.shared:
            return self.net(x), self.net2(x)
        else:
            return self.net(x).chunk(2,dim=1)


""" Models the noise as a gaussian with 0 mean and diagnol covariance. Also
models the mapping from latent space to data space
"""
Decoder = Encoder

class VanillaVAE(nn.Module):
    
    def __init__(self,encoder,decoder,latent_dim,device):
        
        super(VanillaVAE,self).__init__()

        self.enc = encoder
        self.dec = decoder
        self.latent_dim = latent_dim
        self.device = device
        
    def forward(self,x):
        
        # Get the conditional mean and log std
        mu_z, logstd_z = self.enc(x)
        
        # Generate corresponding point in the latent space
        z = torch.randn_like(logstd_z,device=self.device)*logstd_z.exp() + mu_z
        
        # Reconstructed x and logstd for noise
        x_recon, logstd_noise = self.dec(z)

        return x_recon, logstd_noise, mu_z, logstd_z

    def sample(self,k,noisy=False):
        """ Generates k samples in the data space
        """
        z = torch.randn(k,self.latent_dim,device=self.device)
        
        x, logstd_noise = self.dec(z)
        
        if noisy:
            x = torch.randn_like(logstd_noise)*logstd_noise.exp() + x

        return x

    @staticmethod
    def construct(input_dim,latent_dim,hiddens_enc,hiddens_dec,shared_enc,shared_dec,device):
        encoder = Encoder(input_dim,hiddens_enc,latent_dim,shared_enc)
        decoder = Decoder(latent_dim,hiddens_dec,input_dim,shared_dec)

        model = VanillaVAE(encoder,decoder,latent_dim,device)
        model = model.to(device)

        return model

class MultiLayerVAE(nn.Module):

    def __init__(self,encoders,decoders,latent_dim,device):
        
        super(MultiLayerVAE,self).__init__()

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.latent_dim = latent_dim
        self.device = device

    def forward(self,x):
        
        mu_zs = []
        logstd_zs = []

        z = [x]
        for enc in self.encoders:
            mu_z, logstd_z = enc(z[-1])
            z.append(torch.randn_like(logstd_z,device=self.device)*logstd_z.exp() + mu_z)
            
            mu_zs.append(mu_z)
            logstd_zs.append(logstd_z)
        
        mu_zs_prior = []
        logstd_zs_prior = []

        for zz, dec in zip(z[2:],self.decoders[1:]):
            mu_z_prior, logstd_z_prior = dec(zz)

            mu_zs_prior.append(mu_z_prior)
            logstd_zs_prior.append(logstd_z_prior)
        
        mu_zs_prior.append(torch.zeros_like(z[-1],device=self.device))
        logstd_zs_prior.append(torch.zeros_like(z[-1],device=self.device))

        x_recon, logstd_noise = self.decoders[0](z[1])
   
        return x_recon, logstd_noise, z[1:], mu_zs_prior, logstd_zs_prior, mu_zs, logstd_zs
    
    def mapToLatent(self,x):
        
        #TODO Extend this function to provide latent variables form other layers 
        z = x

        for enc in self.encoders:
            mu_z, logstd_z = enc(z)
            z = torch.randn_like(logstd_z,device=self.device)*logstd_z.exp() + mu_z
        
        return z

    @staticmethod
    def construct(input_dim,latent_dim,hiddens_enc,hiddens_dec,shared_enc,shared_dec,device):
        encoders = [ Encoder(input_dim,hiddens_enc[0],latent_dim,shared_enc) ]
        encoders = encoders + [ Encoder(latent_dim,hiddens_enc_i,latent_dim,shared_enc) for hiddens_enc_i in hiddens_enc[1:] ]
        
        decoders = [ Decoder(latent_dim,hiddens_dec[0],input_dim,shared_dec) ]
        decoders = decoders + [ Decoder(latent_dim,hiddens_dec_i,latent_dim,shared_dec) for hiddens_dec_i in hiddens_dec[1:] ]

        model = MultiLayerVAE(encoders,decoders,latent_dim,device)
        model = model.to(device)

        return model
