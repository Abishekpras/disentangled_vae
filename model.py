import torch
import torch.nn as nn
from torch.autograd import Variable
import dist as dist
import math
from numbers import Number


class VAE(nn.Module):
    def __init__(self, z_dim, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=False, mss=False):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        # self.tcvae = tcvae
        self.lamb = 0
        self.beta = 1
        # self.mss = mss
        self.x_dist = dist.Bernoulli()

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks

        self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
        self.decoder = MLPDecoder(z_dim)

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    # def model_sample(self, batch_size=1):
    #     # sample from prior (value will be sampled by guide when computing the ELBO)
    #     prior_params = self._get_prior_params(batch_size)
    #     zs = self.prior_dist.sample(params=prior_params)
    #     # decode the latent code z
    #     x_params = self.decoder.forward(zs)
    #     return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        x = x.view(x.size(0), 1, 64, 64)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), 1, 64, 64)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx

        if self.beta != 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach()

        # # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        # _logqz = self.q_dist.log_density(
        #     zs.view(batch_size, 1, self.z_dim),
        #     z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        # )
        #
        # if not self.mss:
        #     # minibatch weighted sampling
        #     logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
        #     logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        # else:
        #     # minibatch stratified sampling
        #     logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
        #     logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
        #     logqz_prodmarginals = logsumexp(
        #         logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        # if not self.tcvae:
        #     if self.include_mutinfo:
        #         modified_elbo = logpx - self.beta * (
        #             (logqz_condx - logpz) -
        #             self.lamb * (logqz_prodmarginals - logpz)
        #         )
        #     else:
        #         modified_elbo = logpx - self.beta * (
        #             (logqz - logqz_prodmarginals) +
        #             (1 - self.lamb) * (logqz_prodmarginals - logpz)
        #         )
        # else:
        #     if self.include_mutinfo:
        #         modified_elbo = logpx - \
        #             (logqz_condx - logqz) - \
        #             self.beta * (logqz - logqz_prodmarginals) - \
        #             (1 - self.lamb) * (logqz_prodmarginals - logpz)
        #     else:
        #         modified_elbo = logpx - \
        #             self.beta * (logqz - logqz_prodmarginals) - \
        #             (1 - self.lamb) * (logqz_prodmarginals - logpz)
        #
        # return modified_elbo, elbo.detach()


class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

