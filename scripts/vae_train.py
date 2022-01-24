import pandas as pd
from tqdm import tqdm

import torch
# torch.__version__
# '1.4.0'
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F

from micnet.lab import Isolate
from micnet.models import VAE, get_latent, vae_loss


def train(epoch, beta=1, fn='MSE', reduction='sum'):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        # data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar, beta=beta, fn=fn, reduction=reduction)
        loss.backward()

        # writer.add_scalar("Loss/train", loss.item() / len(data), epoch)

        train_loss += loss.item()
        optimizer.step()

        # Log
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)
                )
            )


# README: Access to "main" data restricted.
df = pd.read_csv('data/main.tsv', sep='\t', low_memory=False)
df['date'] = pd.to_datetime(df['date'])

X, y = [], []
cnt = 0
for _, i in tqdm(df.iterrows()):
    try:
        isolate = Isolate(i, expand_mic=True)  # 26 antibiotics
        X.append(isolate.mic.e_log_profile)
        y.append(isolate.species)
    except ValueError:
        cnt += 1
        continue
print(f'{cnt} records were corrupted')
# 40


# Specify model and optimizer
torch.manual_seed(42)
device = torch.device('cuda' if 0 else 'cpu')


# Hyperparameters
epochs = 30
beta = 1    # disentangle more: 1.5
fn = 'MSE'  # MSE, MAE, RMSE (pretty much same as MSE)
lr = 3e-3   # 3e-4
# https://www.jeremyjordan.me/nn-learning-rate/

shuffle = True
batch_size = 64

reduction = 'sum'  
# https://discuss.pytorch.org/t/loss-reduction-sum-vs-mean-when-to-use-each/115641/6


x = [torch.Tensor(i) for i in X]
train_loader = DataLoader(x, batch_size=batch_size, shuffle=shuffle)
model = VAE(len(x[0])).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, epochs + 1):
    train(epoch, beta, fn, reduction)

prefix = 'data'
torch.save(model.state_dict(), f'{prefix}/vae.pt')
model.eval()
z = get_latent(x, model)


selection = ['Escherichia coli', 'Proteus mirabilis', 'Pseudomonas aeruginosa', 'Staphylococcus epidermidis', 'Enterococcus faecalis', 'Staphylococcus aureus', 'Enterococcus faecium', 'Staphylococcus aureus ***MRSA***', 'Escherichia coli (ESBL)', 'Serratia marcescens', 'Enterococcus faecalis ATCC 29212', 'Proteus vulgaris']


abbrev = {
    'Escherichia coli': 'E. coli',
    'Proteus mirabilis': 'Pr. mirabilis',
    'Pseudomonas aeruginosa': 'P. aeruginosa',
    'Staphylococcus epidermidis': 'S. epidermidis',
    'Enterococcus faecalis': 'Ent. faecalis',
    'Staphylococcus aureus': 'S. aureus',
    'Enterococcus faecium': 'Ent. faecium',
    'Staphylococcus aureus ***MRSA***': 'MRSA',
    'Escherichia coli (ESBL)': 'E. coli (ESBL)',
    'Serratia marcescens': 'S. marcescens',
    'Enterococcus faecalis ATCC 29212': 'Ent. faecalis (ATCC 29212)',
    'Proteus vulgaris': 'Pr. vulgaris',
}


with open(f'{prefix}/latent.csv', 'w+') as out:
    out.write('species,c1,c2\n')
    for species, (c1, c2) in zip(y, z):
        if not species in selection:
            species = 'other'
        else:
            species = abbrev[species]

        out.write(f'{species},{c1},{c2}\n')
