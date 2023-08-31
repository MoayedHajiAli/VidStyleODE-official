import numpy as np
import torch
import torch.nn as nn
from src.modules.ablation import utils
from torch.distributions.multivariate_normal import MultivariateNormal
from torchdiffeq import odeint as odeint
import torchvision.models as models
from torch.nn import functional as F
from typing import Tuple, Optional, List
import math

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d)):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def reverse(tensor):
	idx = [i for i in range(tensor.size(0)-1, -1, -1)]
	return tensor[idx]

class ProjNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, n_hidden_layers=1):
        super(ProjNetwork, self).__init__()
        assert (n_hidden_layers > 0), "Need to have a positive number of layers"
        
        module_list = []
        module_list.append(nn.Linear(in_dim, hidden_dim))

        for i in range(n_hidden_layers):
            module_list.append(nn.ReLU())
            module_list.append(nn.Linear(hidden_dim, hidden_dim))
        
        module_list.append(nn.ReLU())
        module_list.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*module_list)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10, hid_channels = 32, kernel_size = 4, hidden_dim = 256):
        # 3dShapes_dataset: latent_dim = 6, hid_channels = 32, kernel_size = 4, hidden_dim = 256
        # fashion_dataset: latent_dim = 16, hid_channels = 32, kernel_size = 4, hidden_dim = 256
        r"""Encoder of the model proposed in [1].
        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent output.
        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(Encoder, self).__init__()

        # Layer parameters
        
        
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels * 2, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels * 2, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels * 2 , hid_channels * 2, kernel_size, **cnn_kwargs)
        elif self.img_size[1] == 128:
            self.conv_64 = nn.Conv2d(hid_channels * 2 , hid_channels * 4, kernel_size, **cnn_kwargs)
            self.conv_128 = nn.Conv2d(hid_channels * 4 , hid_channels * 4, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        #self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

        self.apply(kaiming_init)



    def forward(self, x):


        batch_size = x.size(0)
        #print(x.size())
        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))
        elif self.img_size[1] == 128:
            x = torch.relu(self.conv_64(x))
            x = torch.relu(self.conv_128(x))


        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        #x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar

class EncoderVideo(nn.Module):

    def __init__(self, img_size,
                 latent_dim=6, hid_channels = 32, kernel_size = 4, hidden_dim = 256):

        # 3dShapes_dataset: latent_dim = 6, hid_channels = 32, kernel_size = 4, hidden_dim = 256
        # fashion_dataset: latent_dim = 16, hid_channels = 32, kernel_size = 4, hidden_dim = 256
        # !!!!!!!!!!!!!!!!!!!This model is not used for our final model.!!!!!!!!!!!!!!!!!!!

        super(EncoderVideo, self).__init__()


        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs

        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels * 2, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.reshape = (hid_channels * 2, kernel_size, kernel_size)
            self.conv_64 = nn.Conv2d(hid_channels * 2 , hid_channels * 2, kernel_size, **cnn_kwargs)
        elif self.img_size[1] == 128:

            self.reshape = (hid_channels * 4, 4, 3)
            #print(self.reshape)
            self.conv_64 = nn.Conv2d(hid_channels * 2 , hid_channels * 4, kernel_size, **cnn_kwargs)
            self.conv_128 = nn.Conv2d(hid_channels * 4 , hid_channels * 4, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        #self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)


        self.use_delta_t = False
        """
        # We can try  GRU model for video latent ode

        if self.use_delta_t:
            self.rnn = nn.GRU(hidden_dim + 1, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        """

        self.apply(kaiming_init)

    def reparametrize(self,mu, logvar):
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + std*eps

    def forward(self, x, t):

        batch_size = x.size(0)
        T = x.size(1)
        h = []
        zs = []
        mu_logvar_s=[]
        for i in range(T):
            xi = x[:,i]
        # Convolutional layers with ReLu activations
            xi = torch.relu(self.conv1(xi))
            xi = torch.relu(self.conv2(xi))
            xi = torch.relu(self.conv3(xi))
            if self.img_size[1] == self.img_size[2] == 64:
                xi = torch.relu(self.conv_64(xi))
            elif self.img_size[1] == 128:
                xi = torch.relu(self.conv_64(xi))
                xi = torch.relu(self.conv_128(xi))
        # Fully connected layers with ReLu activations
            xi = xi.view((batch_size, -1))
            hi = torch.relu(self.lin1(xi))
            h.append(hi)

        h = torch.stack(h)
        #h_max = torch.max(h, dim=0)[0].unsqueeze(0)
        #h_mean = torch.mean(h, dim=0).unsqueeze(0)
        #hs = h.clone()
        #hs = h_max.repeat(T,1,1)

        isReverse = False
        if isReverse:
            # we are going backwards in time with
            h = reverse(h)
        
        if self.use_delta_t:
            delta_t = t[1:] - t[:-1]
            if isReverse:
                # we are going backwards in time with
                delta_t = utils.reverse(delta_t)
            delta_t = torch.cat((delta_t, torch.zeros(1).to(delta_t.device)))
            delta_t = delta_t.unsqueeze(1).repeat((1,batch_size)).unsqueeze(-1)
            h = torch.cat((delta_t, h),-1)

        #rnn_out, _ = self.rnn(h.float())
        #print(rnn_out.size())
        #hd_z0 = rnn_out[-1]


        mu_logvar = self.mu_logvar_gen(h.contiguous().view(batch_size * T, -1))
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        z    = self.reparametrize(mu, logvar)

        return z, (mu, logvar)

    def test(self, x, t):

        batch_size = x.size(0)
        T = x.size(1)
        h = []
        zs = []
        mu_logvar_s=[]
        for i in range(T):
            xi = x[:,i]
        # Convolutional layers with ReLu activations
            xi = torch.relu(self.conv1(xi))
            xi = torch.relu(self.conv2(xi))
            xi = torch.relu(self.conv3(xi))
            if self.img_size[1] == self.img_size[2] == 64:
                xi = torch.relu(self.conv_64(xi))
            elif self.img_size[1] == 128:
                xi = torch.relu(self.conv_64(xi))
                xi = torch.relu(self.conv_128(xi))
        # Fully connected layers with ReLu activations
            xi = xi.view((batch_size, -1))
            hi = torch.relu(self.lin1(xi))
            h.append(hi)

        h = torch.stack(h)
        #h_max = torch.max(h, dim=0)[0].unsqueeze(0)
        #h_mean = torch.mean(h, dim=0).unsqueeze(0)
        #hs = h.clone()
        #hs = h_max.repeat(T,1,1)

        isReverse = False
        if isReverse:
            # we are going backwards in time with
            h = reverse(h)
        
        if self.use_delta_t:
            delta_t = t[1:] - t[:-1]
            if isReverse:
                # we are going backwards in time with
                delta_t = utils.reverse(delta_t)
            delta_t = torch.cat((delta_t, torch.zeros(1).to(delta_t.device)))
            delta_t = delta_t.unsqueeze(1).repeat((1,batch_size)).unsqueeze(-1)
            h = torch.cat((delta_t, h),-1)

        #rnn_out, _ = self.rnn(h.float())
        #print(rnn_out.size())
        #hd_z0 = rnn_out[-1]


        mu_logvar = self.mu_logvar_gen(h.contiguous().view(batch_size * T, -1))
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)


        return  mu


class EncoderVideo_LatentODE(nn.Module):
    def __init__(self, img_size, odefunc,
                 static_latent_dim=5, dynamic_latent_dim=1, hid_channels = 32, kernel_size = 4, hidden_dim = 256, use_last_gru_hidden=False,
                 bidirectional_gru=True, num_GRU_layers=4):

        # 3dShapes_dataset: static_latent_dim=5, dynamic_latent_dim=1, hid_channels = 32, kernel_size = 4, hidden_dim = 256
        # fashion_dataset: static_latent_dim=12, dynamic_latent_dim=4,, hid_channels = 32, kernel_size = 4, hidden_dim = 256

        super(EncoderVideo_LatentODE, self).__init__()

        self.static_latent_dim = static_latent_dim
        self.dynamic_latent_dim = dynamic_latent_dim
        self.img_size = img_size
        self.odefunc = odefunc
        self.use_last_gru_hidden = use_last_gru_hidden
        self.num_GRU_layers = num_GRU_layers
        self.bidirectional_gru = bidirectional_gru
        # Shape required to start transpose convs

        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels * 2, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.reshape = (hid_channels * 2, kernel_size, kernel_size)
            self.conv_64 = nn.Conv2d(hid_channels * 2 , hid_channels * 2, kernel_size, **cnn_kwargs)

        elif self.img_size[1] == 128:
            self.reshape = (hid_channels * 4, 4, 3)
            #print(self.reshape)
            self.conv_64 = nn.Conv2d(hid_channels * 2 , hid_channels * 4, kernel_size, **cnn_kwargs)
            self.conv_128 = nn.Conv2d(hid_channels * 4 , hid_channels * 4, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen_s = nn.Linear(hidden_dim, self.static_latent_dim * 2)

        d_hidden_dim = hidden_dim * 2 if bidirectional_gru else hidden_dim
        self.mu_logvar_gen_d = nn.Linear(d_hidden_dim, self.dynamic_latent_dim * 2)

        self.rnn = nn.GRU(hidden_dim + 1, hidden_dim, num_layers=num_GRU_layers, bidirectional=bidirectional_gru, batch_first=False)

        self.apply(kaiming_init)

    def reparametrize(self,mu, logvar):
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + std*eps

    def forward(self, x, t): # x: B x T x C x H x W , t: (B x T)
        batch_size = x.size(0)
        T = x.size(1)

        xi = x.permute(1, 0, 2, 3, 4).contiguous() # T x B x C x H x W
        xi = xi.reshape(T*batch_size, x.shape[2], x.shape[3], x.shape[4]) # T*B x C x H x W

        # Convolutional layers with ReLu activations
        xi = torch.relu(self.conv1(xi))
        xi = torch.relu(self.conv2(xi))
        xi = torch.relu(self.conv3(xi))
        if self.img_size[1] == self.img_size[2] == 64:
            xi = torch.relu(self.conv_64(xi))
        elif self.img_size[1] == 128:
            xi = torch.relu(self.conv_64(xi))
            xi = torch.relu(self.conv_128(xi)) # T*B x C'x H'x W'

        # Fully connected layers with ReLu activations
        xi = xi.view((T * batch_size, -1)) # T*B x C' * H' * W'
        hi = torch.relu(self.lin1(xi)) # T * B x D
        h = hi.view(T, batch_size, -1)  # T x B x D

        h_max = torch.max(h, dim=0)[0] # B x D
        # hs = h_max.repeat(T,1,1) # T x B x D

        isReverse = True # TODO: fix later
        if isReverse:
            # we are going backwards in time with
            h = reverse(h) # T x B x D
        
        self.use_delta_t = True #TODO: fix later
        if self.use_delta_t:
            delta_t = t[1:] - t[:-1] # T - 1 
            if isReverse:
                # we are going backwards in time with
                delta_t = utils.reverse(delta_t)
            delta_t = torch.cat((delta_t, torch.zeros(1).to(delta_t.device))) # T
            delta_t = delta_t.unsqueeze(1).repeat((1,batch_size)).unsqueeze(-1) # concat time. Why not pos encoding?  # T x B x 1
            h = torch.cat((delta_t, h),-1) # T x B x (D+1)

        rnn_out, last_hidden = self.rnn(h.float()) # this uses static and dynamic features
        if self.use_last_gru_hidden:
            # TODO: fix 
            if self.bidirectional_gru:
                hd_z0 = torch.cat((last_hidden[0], last_hidden[1]), -1)
            else:
                hd_z0 = last_hidden[0] # TODO: test this when switching to bidirectonal
        else:
            hd_z0 = rnn_out[-1] # B x D
        
        mu_logvar_d0 = self.mu_logvar_gen_d(hd_z0) # B x D' * 2
        mu_d0, logvar_d0 = mu_logvar_d0.view(-1, self.dynamic_latent_dim, 2).unbind(-1)

        zd0 = self.reparametrize(mu_d0, logvar_d0) # B x D' 
        zd0 = zd0.to(torch.float64)
        zdt = odeint(self.odefunc,zd0,t,method='dopri5') # aren't we here solving the ODE jointly for all videos? Shouldn't we separate them?
        zdt = zdt.to(torch.float32) # T x B x D
        zdt = zdt.view(batch_size * T, -1) # T*B x D 

        # Question: why using hs and not h_max? Isn't redundent? RIP
        mu_logvars = self.mu_logvar_gen_s(h_max) # B x 2 * D
        mus, logvars = mu_logvars.view(-1, self.static_latent_dim, 2).unbind(-1) # B x D 
        zs = self.reparametrize(mus, logvars) # B x D 
        zs = zs.unsqueeze(0).repeat(T, 1, 1).view(T * batch_size, -1)

        return zs, zdt, (mus, logvars), (mu_d0, logvar_d0)

    def test(self, x, t):

        batch_size = x.size(0)
        T = x.size(1)
        h = []
        zs = []
        mu_logvar_s=[]
        for i in range(T):
            xi = x[:,i]
        # Convolutional layers with ReLu activations
            xi = torch.relu(self.conv1(xi))
            xi = torch.relu(self.conv2(xi))
            xi = torch.relu(self.conv3(xi))
            if self.img_size[1] == self.img_size[2] == 64:
                xi = torch.relu(self.conv_64(xi))
            elif self.img_size[1] == 128:
                xi = torch.relu(self.conv_64(xi))
                xi = torch.relu(self.conv_128(xi))
        # Fully connected layers with ReLu activations
            xi = xi.view((batch_size, -1))
            hi = torch.relu(self.lin1(xi))
            h.append(hi)

        h = torch.stack(h)

        h_max = torch.max(h, dim=0)[0].unsqueeze(0)
        #h_mean = torch.mean(h, dim=0).unsqueeze(0)
        #hs = h.clone()
        hs = h_max.repeat(T,1,1)


        isReverse = True
        if isReverse:
            # we are going backwards in time with
            h = reverse(h)
        
        if self.use_delta_t:
            delta_t = t[1:] - t[:-1]
            if isReverse:
                # we are going backwards in time with
                delta_t = utils.reverse(delta_t)
            delta_t = torch.cat((delta_t, torch.zeros(1).to(delta_t.device)))
            delta_t = delta_t.unsqueeze(1).repeat((1,batch_size)).unsqueeze(-1)
            h = torch.cat((delta_t, h),-1)

        rnn_out, _ = self.rnn(h.float())
        #print(rnn_out.size())
        hd_z0 = rnn_out[-1]
        #hd_z0 = torch.tanh(hd_z0)
        mu_logvar_d0 = self.mu_logvar_gen_d(hd_z0)
        mu_d0, logvar_d0 = mu_logvar_d0.view(-1, self.dynamic_latent_dim, 2).unbind(-1)

        zdt = odeint(self.odefunc,mu_d0, t,method='dopri5') 
        #print(zdt.size())
        zdt = zdt.view(batch_size * T, -1)

        mu_logvars = self.mu_logvar_gen_s(hs.contiguous().view(batch_size * T, -1))
        mus, logvars = mu_logvars.view(-1, self.static_latent_dim, 2).unbind(-1)

        return mus, zdt

    def test_ode(self, x, t, t_tar):

        batch_size = x.size(0)
        T = x.size(1)
        h = []
        zs = []
        mu_logvar_s=[]
        for i in range(T):
            xi = x[:,i]
        # Convolutional layers with ReLu activations
            xi = torch.relu(self.conv1(xi))
            xi = torch.relu(self.conv2(xi))
            xi = torch.relu(self.conv3(xi))
            if self.img_size[1] == self.img_size[2] == 64:
                xi = torch.relu(self.conv_64(xi))
            elif self.img_size[1] == 128:
                xi = torch.relu(self.conv_64(xi))
                xi = torch.relu(self.conv_128(xi))
        # Fully connected layers with ReLu activations
            xi = xi.view((batch_size, -1))
            hi = torch.relu(self.lin1(xi))
            h.append(hi)

        h = torch.stack(h)

        h_max = torch.max(h, dim=0)[0].unsqueeze(0)
        #h_mean = torch.mean(h, dim=0).unsqueeze(0)
        #hs = h.clone()
        hs = h_max.repeat(t_tar.size(0),1,1)


        isReverse = True
        if isReverse:
            # we are going backwards in time with
            h = reverse(h)
        
        if self.use_delta_t:
            delta_t = t[1:] - t[:-1]
            if isReverse:
                # we are going backwards in time with
                delta_t = utils.reverse(delta_t)
            delta_t = torch.cat((delta_t, torch.zeros(1).to(delta_t.device)))
            delta_t = delta_t.unsqueeze(1).repeat((1,batch_size)).unsqueeze(-1)
            h = torch.cat((delta_t, h),-1)

        rnn_out, _ = self.rnn(h.float())
        #print(rnn_out.size())
        hd_z0 = rnn_out[-1]
        #hd_z0 = torch.tanh(hd_z0)
        mu_logvar_d0 = self.mu_logvar_gen_d(hd_z0)
        mu_d0, logvar_d0 = mu_logvar_d0.view(-1, self.dynamic_latent_dim, 2).unbind(-1)

        zdt = odeint(self.odefunc,mu_d0, t_tar,method='dopri5') 
        #print(zdt.size())
        zdt = zdt.view(batch_size * t_tar.size(0), -1)

        mu_logvars = self.mu_logvar_gen_s(hs.contiguous().view(batch_size * t_tar.size(0), -1))
        mus, logvars = mu_logvars.view(-1, self.static_latent_dim, 2).unbind(-1)

        return mus, zdt

class Decoder(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):

        super(Decoder, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        #self.reshape = (hid_channels * 2, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        #self.lin2 = nn.Linear(hidden_dim, hidden_dim)


        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.reshape = (hid_channels * 2, kernel_size, kernel_size)
            self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))
            self.convT_64 = nn.ConvTranspose2d(hid_channels * 2, hid_channels* 2, kernel_size, **cnn_kwargs)
        elif self.img_size[1] == 128:
            self.reshape = (hid_channels * 4, 4, 3)
            self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))
            self.convT_128 = nn.ConvTranspose2d(hid_channels * 4, hid_channels* 4, kernel_size, **cnn_kwargs)
            self.convT_64 = nn.ConvTranspose2d(hid_channels * 4, hid_channels* 2, kernel_size, **cnn_kwargs)
        self.convT1 = nn.ConvTranspose2d(hid_channels * 2, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)
        self.apply(kaiming_init)
        
    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        #x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        elif self.img_size[1] == 128:
            x = torch.relu(self.convT_128(x))
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x

# This code was taken directly from github reposiroty of torchdiffeq python package
# https://github.com/rtqichen/torchdiffeq
class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=1, nhidden=10):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden, dtype=torch.float64)
        self.fc2 = nn.Linear(nhidden, nhidden, dtype=torch.float64)
        self.fc21 = nn.Linear(nhidden, nhidden, dtype=torch.float64)
        # self.fc22 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim, dtype=torch.float64)
        self.nfe = 0
        #self.apply(weights_init)
        self.latent_dim = latent_dim

    def forward(self, t, x):
        self.nfe += 1
        #print(self.nfe)
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc21(out)
        out = self.elu(out)
        # out = self.fc22(out)
        # out = self.elu(out)
        out = self.fc3(out)
        return out
    
    def callback_step(self, t0, y0, dt):
        pass


# This stands for the only text conditioned model.
class AdaIn(nn.Module):
    '''
    adaptive instance normalization
    '''
    def __init__(self, nchannel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(nchannel)
        
    def forward(self, img_feat, w):
        factor, bias = w.chunk(2, 1)
        result = self.norm(img_feat)
        result = result * factor + bias  
        return result

class MFMOD(nn.Module):
    '''
    Multi-Feature Modulation Block
    '''
    def __init__(self, nchannel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(nchannel)
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(nchannel), requires_grad=True)
    def forward(self, img_feat, w1, w2):
        factor1, bias1 = w1.chunk(2, 1)
        factor2, bias2 = w2.chunk(2, 1)
        gamma_alpha = torch.sigmoid(self.blending_gamma)
        beta_alpha = torch.sigmoid(self.blending_beta)
        #beta_alpha = 1.0 #this improves text conditioning.
        #print(gamma_alpha)
        #print(beta_alpha)
        factor = gamma_alpha * factor1 + (1 - gamma_alpha) * factor2
        bias = beta_alpha * bias1 + (1 - beta_alpha) * bias2
        added_noise = (torch.randn(img_feat.shape[0], img_feat.shape[3], img_feat.shape[2], 1).to(img_feat.device) * self.noise_var).transpose(1, 3)
        result = self.norm(img_feat + added_noise)
        result = result * factor + bias  
        return result

# This stands for the only text conditioned model.
class ResidualAdaInBlock(nn.Module):
	def __init__(self, fsize=512, fmapsize=256):
		super(ResidualAdaInBlock, self).__init__()

		self.style1   = nn.Linear(fmapsize, fsize * 2)
		self.style2   = nn.Linear(fmapsize, fsize * 2)
		self.adain    = AdaIn(fsize)
		self.lrelu    = nn.LeakyReLU(0.2)

		self.conv1 = nn.Conv2d(fsize, fsize, 3, padding=1)
		self.conv2 = nn.Conv2d(fsize, fsize, 3, padding=1)

	def forward(self, x, latent_w):
		out = self.conv1(x)
		style1 = self.style1(latent_w).unsqueeze(2).unsqueeze(3)
		out = self.adain(out, style1)
		out = self.lrelu(out)
		out = self.conv2(out)
		style2 = self.style2(latent_w).unsqueeze(2).unsqueeze(3)
		out = self.adain(out, style2)
		# out = self.lrelu(out)

		return self.lrelu(x + out)
		#return self.lrelu(out)

class ResidualMFMODBlock(nn.Module):
	def __init__(self, fsize=512, fmapsize=256):
		super(ResidualMFMODBlock,  self).__init__()

		self.style1_w1   = nn.Linear(512, fsize * 2)
		self.style2_w1   = nn.Linear(512, fsize * 2)

		self.style1_w2   = nn.Linear(fmapsize, fsize * 2)
		self.style2_w2   = nn.Linear(fmapsize, fsize * 2)

		self.adain    = MFMOD(fsize)
		self.lrelu    = nn.LeakyReLU(0.2)

		self.conv1 = nn.Conv2d(fsize, fsize, 3, padding=1)
		self.conv2 = nn.Conv2d(fsize, fsize, 3, padding=1)

	def forward(self, x, latent_w1, latent_w2):
		out = self.conv1(x)
		style1_w1 = self.style1_w1(latent_w1).unsqueeze(2).unsqueeze(3)
		style1_w2 = self.style1_w2(latent_w2).unsqueeze(2).unsqueeze(3)

		out = self.adain(out, style1_w1, style1_w2)
		out = self.lrelu(out)
		out = self.conv2(out)
		style2_w1 = self.style2_w1(latent_w1).unsqueeze(2).unsqueeze(3)
		style2_w2 = self.style2_w2(latent_w2).unsqueeze(2).unsqueeze(3)
		out = self.adain(out, style2_w1, style2_w2)

		return self.lrelu(x + out)
	    

# This stands for the only text conditioned model.    
class Generator(nn.Module):
	def __init__(self, fsize=64):
		super(Generator, self).__init__()
		self.encoder = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(3, fsize, 7, padding=0),
			nn.InstanceNorm2d(fsize),
			nn.ReLU(inplace=True),

			nn.Conv2d(fsize, fsize*2, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm2d(fsize*2),
			nn.ReLU(inplace=True),

			nn.Conv2d(fsize*2, fsize*4, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm2d(fsize*4),
			nn.ReLU(inplace=True),

			nn.Conv2d(fsize*4, fsize*8, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm2d(fsize*8),
			nn.ReLU(inplace=True),

		)

		self.res_block1 = ResidualAdaInBlock(fsize=fsize*8, fmapsize=512)
		self.res_block2 = ResidualAdaInBlock(fsize=fsize*8, fmapsize=512)
		self.res_block3 = ResidualAdaInBlock(fsize=fsize*8, fmapsize=512)
		self.res_block4 = ResidualAdaInBlock(fsize=fsize*8, fmapsize=512)
		self.res_block5 = ResidualAdaInBlock(fsize=fsize*8, fmapsize=512)

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(fsize*8, fsize*4, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.InstanceNorm2d(fsize*4),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(fsize*4, fsize*2, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.InstanceNorm2d(fsize*2),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(fsize*2, fsize, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.InstanceNorm2d(fsize),
			nn.ReLU(inplace=True),

			nn.ReflectionPad2d(3),
			nn.Conv2d(fsize, 3, kernel_size=7, padding=0),
			nn.Tanh()

		)


		self.apply(kaiming_init)

	def forward(self, img, latent_w):
		img_feat = self.encoder(img)
		out1 = self.res_block1(img_feat, latent_w)
		out2 = self.res_block2(out1, latent_w)
		out3 = self.res_block3(out2, latent_w)
		out4 = self.res_block4(out3, latent_w)
		out5 = self.res_block5(out4, latent_w)

		output = self.decoder(out5)

		return output


# Proposed TraNet with MFMOD blocks   
class Generator2(nn.Module):
	def __init__(self, fsize=64):
		super(Generator2, self).__init__()
		self.encoder = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(3, fsize, 7, padding=0),
			nn.InstanceNorm2d(fsize), # Question: why there is instance normalization in the encoder? TODO: Check Pic2picHD model. 
			nn.ReLU(inplace=True),

			nn.Conv2d(fsize, fsize*2, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm2d(fsize*2),
			nn.ReLU(inplace=True),

			nn.Conv2d(fsize*2, fsize*4, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm2d(fsize*4),
			nn.ReLU(inplace=True),

			nn.Conv2d(fsize*4, fsize*8, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm2d(fsize*8),
			nn.ReLU(inplace=True),

		)

		self.res_block1 = ResidualMFMODBlock(fsize=fsize*8, fmapsize=256)
		self.res_block2 = ResidualMFMODBlock(fsize=fsize*8, fmapsize=256)
		self.res_block3 = ResidualMFMODBlock(fsize=fsize*8, fmapsize=256)
		self.res_block4 = ResidualMFMODBlock(fsize=fsize*8, fmapsize=256)
		self.res_block5 = ResidualMFMODBlock(fsize=fsize*8, fmapsize=256)

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(fsize*8, fsize*4, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.InstanceNorm2d(fsize*4),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(fsize*4, fsize*2, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.InstanceNorm2d(fsize*2),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(fsize*2, fsize, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.InstanceNorm2d(fsize),
			nn.ReLU(inplace=True),

			nn.ReflectionPad2d(3),
			nn.Conv2d(fsize, 3, kernel_size=7, padding=0),
			nn.Tanh()

		)


		self.apply(kaiming_init)

	def forward(self, img, txt_feat, latent_w):
		img_feat = self.encoder(img)
		out1 = self.res_block1(img_feat, txt_feat, latent_w)
		out2 = self.res_block2(out1, txt_feat, latent_w)
		out3 = self.res_block3(out2, txt_feat, latent_w)
		out4 = self.res_block4(out3, txt_feat, latent_w)
		out5 = self.res_block5(out4, txt_feat, latent_w)

		output = self.decoder(out5)

		return output

# Auxiliary Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, text_size,
                 latent_dim=10):
        self.latent_dim = latent_dim
        super(TextEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(text_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, latent_dim * 2)
        )

        self.apply(kaiming_init)



    def forward(self, x):


        batch_size = x.size(0)
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar





# Normalization on every element of input vector
# These codes are adapted from the pytorch implementation of StyleGAN2
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class EqualizedWeight(nn.Module):

	def __init__(self, shape: List[int]):
		super().__init__()
		self.c = 1 / math.sqrt(np.prod(shape[1:]))
		self.weight = nn.Parameter(torch.randn(shape))

	def forward(self):
		return self.weight * self.c


class EqualizedLinear(nn.Module):
	def __init__(self, in_features: int, out_features: int, bias: float = 0.):
		super().__init__()
		self.weight = EqualizedWeight([out_features, in_features])
		self.bias = nn.Parameter(torch.ones(out_features) * bias)

	def forward(self, x: torch.Tensor):
		return F.linear(x, self.weight(), bias=self.bias)


class MappingNetworkVAE(nn.Module):
	def __init__(self, input_dim=8,  fsize = 256):
		super(MappingNetworkVAE, self).__init__()

		self.linvae   = nn.Sequential(EqualizedLinear(input_dim, fsize),
					      nn.LeakyReLU(0.2, inplace=True))

		self.mapping = nn.Sequential( 
					      EqualizedLinear(fsize, fsize),
					      nn.LeakyReLU(0.2, inplace=True),
					      EqualizedLinear(fsize, fsize),
					      nn.LeakyReLU(0.2, inplace=True),
					      EqualizedLinear(fsize, fsize),
					      nn.LeakyReLU(0.2, inplace=True))

		self.pixel_norm = PixelNorm()
		self.apply(kaiming_init)


	def forward(self, vae_feat):
                vae_feat = self.pixel_norm(vae_feat)
                out = self.linvae(vae_feat)
                out = self.mapping(out)
                return out



class MultiscaleDiscriminatorPixpixHD(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminatorPixpixHD, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminatorPix2pixHD(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input, latentw):        
        num_D = self.num_D
        result = []
        latentw = latentw.unsqueeze(2).unsqueeze(3)
        h = input.size(2)
        w = input.size(3)
        latentw = latentw.expand(-1,-1,h,w)
        input = torch.cat((input, latentw), 1)
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminatorPix2pixHD(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminatorPix2pixHD, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)  

class Vgg19(torch.nn.Module):
	def __init__(self, requires_grad=False):
		super().__init__()
		vgg_pretrained_features = models.vgg19(pretrained=True).features
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()

		for x in range(2):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(2, 7):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(7, 12):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(12, 21):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(21, 30):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, X):
		h_relu1 = self.slice1(X)
		h_relu2 = self.slice2(h_relu1)
		h_relu3 = self.slice3(h_relu2)
		h_relu4 = self.slice4(h_relu3)
		h_relu5 = self.slice5(h_relu4)

		out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

		return out

class MultiscaleDiscriminatorPixpixHDMFMOD(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminatorPixpixHDMFMOD, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminatorPix2pixHDMFMOD(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD)# netD.model changed to netD

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input, txt_feat, vae_feat):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            #print(model)
            return [model(input, txt_feat, vae_feat)]

    def forward(self, input, txt_feat, vae_feat):        
        num_D = self.num_D
        result = []
        h = input.size(2)
        w = input.size(3)
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))


            result.append(self.singleD_forward(model, input_downsampled, txt_feat, vae_feat))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminatorPix2pixHDMFMOD(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminatorPix2pixHDMFMOD, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        self.res_block = ResidualMFMODBlock(fsize=ndf*8, fmapsize=256)

        nf_prev = nf
        nf = min(nf * 2, 512)

        self.classifier = nn.Sequential(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw))

        if use_sigmoid:
            self.classifier += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input, txt_feat, vae_feat):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            out =  self.model(input)
            out =  self.res_block(out, txt_feat, vae_feat)
            return self.classifier(out) 

# This stands for the only text conditioned model.
class MultiscaleDiscriminatorPixpixHDAdaINText(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminatorPixpixHDAdaINText, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminatorPix2pixHDAdaINText(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD)# netD.model changed to netD

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input, txt_feat):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            #print(model)
            return [model(input, txt_feat)]

    def forward(self, input, txt_feat):        
        num_D = self.num_D
        result = []
        h = input.size(2)
        w = input.size(3)
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))


            result.append(self.singleD_forward(model, input_downsampled, txt_feat))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminatorPix2pixHDAdaINText(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminatorPix2pixHDAdaINText, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        self.res_block = ResidualAdaInBlock(fsize=ndf*8, fmapsize=512)

        nf_prev = nf
        nf = min(nf * 2, 512)
        self.classifier = nn.Sequential(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw))

        if use_sigmoid:
            self.classifier += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input, txt_feat):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            out =  self.model(input)
            out =  self.res_block(out, txt_feat)
            return self.classifier(out) 
