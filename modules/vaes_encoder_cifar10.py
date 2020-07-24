from torch import nn


class Reshape(nn.Module):
    def __init__(self, shape, **kwargs):
        self.shape = shape
        super(Reshape, self).__init__(**kwargs)

    def forward(self, input):
        return input.view(self.shape)
    
class Net(nn.Module):
    """ A base class for both generator and the discriminator.
    Provides a common weight initialization scheme.

    """

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            if "Conv" in classname:
                m.weight.data.normal_(0.0, 0.02)

            elif "BatchNorm" in classname:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        return x

    
class Encoder(Net):

    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

            
        self.fc1 = nn.Linear(in_features=4096, out_features=512)
        self.r1  = nn.LeakyReLU(0.2, inplace=False)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.r2 =  nn.LeakyReLU(0.2, inplace=False)
        
        
        self.mu_projection = nn.Linear(in_features=256, out_features=z_dim)
        self.logsigmasq_projection = nn.Linear(
            in_features=256, out_features=z_dim)

        self.weights_init()

    def forward(self, x):

        x = self.net(x)
        x = x.view(x.shape[0], -1)

        x = self.r1(self.fc1(x))
        x = self.r2(self.fc2(x))
        
        mu = self.mu_projection(x)
        logsigmasq = self.logsigmasq_projection(x)
        return mu, logsigmasq