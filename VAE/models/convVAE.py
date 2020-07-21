import torch
import torch.nn as nn
from .helper import activations
import numpy as np

class Encoder(nn.Module):
    """ Assumes shared parameters between mean and logstd for latent space
    """
    def __init__(self,input_size,conv_config,latent_dim):
        """
        Args:
            input_size (tuple(int,int)):
            input_channels (int)
            conv_config (list):
                {
                    'channel': (int),
                    'kernel': (int),
                    'stride': (int),
                    'act': (activation) = nn.ReLU()
                    'bn': (bool) = False
                }
            latent_dim (int)
        """
        super(Encoder, self).__init__()
        
        in_channels = [input_size[2]] + [d['channel'] for d in conv_config[:-1]]
        
        layers = [ l for indim, d in zip(in_channels,conv_config) for l in \
            ((nn.Conv2d(indim,d['channel'],d['kernel'],d['stride'],\
            d['kernel']//2), nn.BatchNorm2d(d['channel']),\
            activations[d.get('act','relu')]()) if d.get('bn',False) else \
            (nn.Conv2d(indim,d['channel'],d['kernel'],d['stride'],
            d['kernel']//2), activations[d.get('act','relu')]()))]
        
        self.conv = nn.Sequential(*layers)
        factor = np.product([ d['stride'] for d in conv_config ])

        conv_out_dim = input_size[0] // factor * input_size[1] // factor \
            * conv_config[-1]['channel']

        self.fc = nn.Linear( conv_out_dim, 2*latent_dim)

    def forward(self,x):
        
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        mu_z, logstd_z = x.chunk(2,dim=1)

        return mu_z, logstd_z

class Decoder(nn.Module):
    
    def __init__(self,output_size,base_channel,conv_config,latent_dim,device):
       
        super(Decoder,self).__init__()
        
        self.device = device
        in_channels = [base_channel] + [d['channel'] for d in conv_config[:-1]]
        
        factor = np.product([ d['stride'] for d in conv_config ])
        self.base_size = (base_channel, output_size[0]//factor, 
            output_size[1]//factor )

        self.fc = nn.Linear(latent_dim,np.product(self.base_size))
        
        #FIXME Need to correct the padding calculation
        layers = [ l for indim, d in zip(in_channels,conv_config) for l in \
            ((nn.BatchNorm2d(d['channel']), activations[d.get('act','relu')](),
            nn.ConvTranspose2d(indim,d['channel'],d['kernel'],d['stride'],1)) 
            if d.get('bn',False) else (activations[d.get('act','relu')](),
            nn.ConvTranspose2d(indim,d['channel'],d['kernel'],d['stride'],1)))] 
        
        layers.append(activations['relu']())
        layers.append(nn.Conv2d(conv_config[-1]['channel'],output_size[2],3,1,1))
        
        self.deconv = nn.Sequential(*layers)

    def forward(self,z):
        
        z = self.fc(z)
        z = z.reshape(-1,*self.base_size)
        x_recon = self.deconv(z)
        
        # implicitely assume variance of 1
        logstd_noise = torch.zeros_like(x_recon,device=self.device)

        return x_recon, logstd_noise

class ConvVAE(nn.Module):

    def __init__(self,encoder,decoder,latent_dim,device):
        
        super(ConvVAE,self).__init__()

        self.enc = encoder
        self.dec = decoder
        self.latent_dim = latent_dim
        self.device = device

    def forward(self,x):
        
        mu_z, logstd_z = self.enc(x)
        
        z = torch.randn_like(logstd_z,device=self.device)*logstd_z.exp() + mu_z

        x_recon, logstd_noise = self.dec(z)

        return x_recon, logstd_noise, mu_z,logstd_z

    def sample(self,k,noisy=False):
        
        z = torch.randn(k,self.latent_dim,device=self.device)

        x, logstd_noise = self.dec(z)

        if noisy:
            x = torch.randn_like(logstd_noise,device=self.device)*logstd_noise.exp() + x

        return x
    
    def mapToLatent(self,x):
        
        mu_z, logstd_z = self.enc(x)
        
        z = torch.randn_like(logstd_z,device=self.device)*logstd_z.exp() + mu_z
        
        return z

    @staticmethod
    def construct(input_size,base_channel,config_enc,config_dec,latent_dim,device):
        
        encoder = Encoder(input_size,config_enc,latent_dim)
        decoder = Decoder(input_size,base_channel,config_dec,latent_dim,device)

        model = ConvVAE(encoder,decoder,latent_dim,device)
        model = model.to(device)

        return model
