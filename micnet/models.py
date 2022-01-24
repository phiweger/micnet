from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    '''
    Probably adopted from

    https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/

    > In the ideal case you latent representation (𝜇 or z) will contain meaningful information, these are the ones I would extract -- https://stats.stackexchange.com/questions/483785/variational-autoencoder-vae-latent-features

    > We observe that sampling a stochastic encoder in a Gaussian VAE can be interpreted as simply injecting noise into the input of a deterministic decoder. -- https://arxiv.org/abs/1903.12436

    - https://github.com/AntixK/PyTorch-VAE
    - https://github.com/taldatech/soft-intro-vae-pytorch
    - https://github.com/zalandoresearch/pytorch-vq-vae
    '''
    def __init__(self, input_size):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_size, 10)
        self.fc21 = nn.Linear(10, 2)
        self.fc22 = nn.Linear(10, 2)
        self.fc3 = nn.Linear(2, 10)
        self.fc4 = nn.Linear(10, input_size)
        # TODO: Add softmax here so we limit the latent space to [0, 1]?

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        # We don't want the sigmoid if we generate sth bc/ the output will
        # be scaled to [0, 1], which is not ideal for eg (log) MIC values.
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)  # x.view(-1, 784)
        z = self.reparameterize(mu, logvar)
        # TODO: z should be the latent representation we want
        return self.decode(z), mu, logvar



def get_latent(data, model):
    zz = []
    # for i in tqdm(data):
    for i in data:
        z = model.reparameterize(*model.encode(i))
        # Turn tensor back into np array (no grad)
        zz.append(z.detach().numpy())
    return zz


def vae_loss(recon_x, x, mu, logvar, beta=1, fn='BCE', reduction='sum'):
    '''
    - https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    - https://discuss.pytorch.org/t/loss-reduction-sum-vs-mean-when-to-use-each/115641/6
    - http://zerospectrum.com/2019/06/02/mae-vs-mse-vs-rmse/

    BCE if we add the sigmoid to the VAE model
    '''
    if fn == 'BCE':
        loss = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    
    elif fn == 'MSE':
        loss = F.mse_loss(recon_x, x, reduction=reduction)  # sum, mean
    
    elif fn == 'RMSE':
        # https://discuss.pytorch.org/t/rmse-loss-function/16540
        loss = torch.sqrt(F.mse_loss(recon_x, x, reduction=reduction))

    elif fn == 'MAE':
        loss = F.l1_loss(recon_x, x, reduction=reduction)  # sum, mean

    else:
        raise ValueError('Loss fn not implemented')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # return loss + KLD
    # beta VAE
    # https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html#beta-vae
    return loss + (beta * KLD)  # beta-VAE


def encode(v, model):
        '''
        n = []
        for _ in range(100):
            n.append(
                model.reparameterize(
                    *model.encode(torch.Tensor(p))).detach().numpy())
        lat = np.mean(n, axis=0)
        '''
        model.eval()
        return model.reparameterize(
            *model.encode(torch.Tensor(v))).detach().numpy()


def decode(emb, model):
    '''
    # Decode
    # Predict single image
    a2[:1]
    emb = model_ft(a1[:1])
    p = np.exp(model.decode(emb).detach().numpy())[0]
    p = [round(float(i), 4) for i in p]
    [print(i, j) for i, j in zip(isolate.mic.abx, p)]
    '''
    model.eval()
    return model.decode(emb).detach().numpy()

