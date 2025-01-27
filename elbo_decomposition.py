import os
import math
from numbers import Number
from tqdm import tqdm
import torch
from torch.autograd import Variable

import dist as dist


def estimate_entropies(qz_samples, qz_params, q_dist):

    # Only take a sample subset of the samples
    qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:10000]))

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert(nparams == q_dist.nparams)
    assert(K == qz_params.size(1))

    marginal_entropies = torch.zeros(K)
    joint_entropy = torch.zeros(1)

    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        marginal_entropies += (math.log(N) - logsumexp(logqz_i, dim=0, keepdim=False).data).sum(1)
        # computes - log q(z) summed over minibatch
        logqz = logqz_i.sum(1)  # (N, S)
        joint_entropy += (math.log(N) - logsumexp(logqz, dim=0, keepdim=False).data).sum(0)
        pbar.update(batch_size)
    pbar.close()

    marginal_entropies /= S
    joint_entropy /= S

    return marginal_entropies, joint_entropy


def logsumexp(value, dim=None, keepdim=False):
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


def analytical_NLL(qz_params, q_dist, prior_dist, qz_samples=None):
    pz_params = Variable(torch.zeros(1).type_as(qz_params.data).expand(qz_params.size()), volatile=True)

    nlogqz_condx = q_dist.NLL(qz_params).mean(0)
    nlogpz = prior_dist.NLL(pz_params, qz_params).mean(0)
    return nlogqz_condx, nlogpz


def elbo_decomposition(vae, dataset_loader):
    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    S = 1                            # number of latent variable samples
    nparams = vae.q_dist.nparams

    print('Computing q(z|x) distributions.')
    # compute the marginal q(z_j|x_n) distributions
    qz_params = torch.Tensor(N, K, nparams)
    n = 0
    logpx = 0
    for xs in dataset_loader:
        batch_size = xs.size(0)
        xs = Variable(xs.view(batch_size, -1, 64, 64), volatile=True)
        z_params = vae.encoder.forward(xs).view(batch_size, K, nparams)
        qz_params[n:n + batch_size] = z_params.data
        n += batch_size

        # estimate reconstruction term
        for _ in range(S):
            z = vae.q_dist.sample(params=z_params)
            x_params = vae.decoder.forward(z)
            logpx += vae.x_dist.log_density(xs, params=x_params).view(batch_size, -1).data.sum()
    # Reconstruction term
    logpx = logpx / (N * S)

    qz_params = Variable(qz_params, volatile=True)

    print('Sampling from q(z).')
    # sample S times from each marginal q(z_j|x_n)
    qz_params_expanded = qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)
    qz_samples = vae.q_dist.sample(params=qz_params_expanded)
    qz_samples = qz_samples.transpose(0, 1).contiguous().view(K, N * S)

    print('Estimating entropies.')
    marginal_entropies, joint_entropy = estimate_entropies(qz_samples, qz_params, vae.q_dist)

    if hasattr(vae.q_dist, 'NLL'):
        nlogqz_condx = vae.q_dist.NLL(qz_params).mean(0)
    else:
        nlogqz_condx = - vae.q_dist.log_density(qz_samples,
            qz_params_expanded.transpose(0, 1).contiguous().view(K, N * S)).mean(1)

    if hasattr(vae.prior_dist, 'NLL'):
        pz_params = vae._get_prior_params(N * K).contiguous().view(N, K, -1)
        nlogpz = vae.prior_dist.NLL(pz_params, qz_params).mean(0)
    else:
        nlogpz = - vae.prior_dist.log_density(qz_samples.transpose(0, 1)).mean(0)

    # nlogqz_condx, nlogpz = analytical_NLL(qz_params, vae.q_dist, vae.prior_dist)
    nlogqz_condx = nlogqz_condx.data
    nlogpz = nlogpz.data

    # Independence term
    # KL(q(z)||prod_j q(z_j)) = log q(z) - sum_j log q(z_j)
    dependence = (- joint_entropy + marginal_entropies.sum())[0]

    # Information term
    # KL(q(z|x)||q(z)) = log q(z|x) - log q(z)
    information = (- nlogqz_condx.sum() + joint_entropy)[0]

    # Dimension-wise KL term
    # sum_j KL(q(z_j)||p(z_j)) = sum_j (log q(z_j) - log p(z_j))
    dimwise_kl = (- marginal_entropies + nlogpz).sum()

    # Compute sum of terms analytically
    # KL(q(z|x)||p(z)) = log q(z|x) - log p(z)
    analytical_cond_kl = (- nlogqz_condx + nlogpz).sum()

    print('Dependence: {}'.format(dependence))
    print('Information: {}'.format(information))
    print('Dimension-wise KL: {}'.format(dimwise_kl))
    print('Analytical E_p(x)[ KL(q(z|x)||p(z)) ]: {}'.format(analytical_cond_kl))
    print('Estimated  ELBO: {}'.format(logpx - analytical_cond_kl))

    return logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpt', required=True)
    parser.add_argument('-save', type=str, default='.')
    parser.add_argument('-gpu', type=int, default=0)
    args = parser.parse_args()

    def load_model_and_dataset(checkpt_filename):
        checkpt = torch.load(checkpt_filename)
        args = checkpt['args']
        state_dict = checkpt['state_dict']

        # backwards compatibility
        if not hasattr(args, 'conv'):
            args.conv = False

        from model import VAE, setup_data_loaders

        # model
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
        vae = VAE(z_dim=args.latent_dim, use_cuda=False, prior_dist=prior_dist, q_dist=q_dist, conv=args.conv)
        vae.load_state_dict(state_dict, strict=False)
        vae.eval()

        # dataset loader
        loader = setup_data_loaders(args, use_cuda=False)
        return vae, loader

    # torch.cuda.set_device(args.gpu)
    vae, dataset_loader = load_model_and_dataset(args.checkpt)
    logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
        elbo_decomposition(vae, dataset_loader)
    torch.save({
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': marginal_entropies,
        'joint_entropy': joint_entropy
    }, os.path.join(args.save, 'elbo_decomposition.pth'))
