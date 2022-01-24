from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
device = torch.device('cuda' if 0 else 'cpu')

from micnet.lab import Isolate
from micnet.models import VAE, get_latent


# Load model
# Load VAE model
folder = '../data'
name = 'vae.pt'
dim = 26  # expanded MIC
model = VAE(dim).to(device)
state = torch.load(f'{folder}/{name}')
model.load_state_dict(state)
# <All keys matched successfully>


# Load data
# README: Data "main" is restricted.
df = pd.read_csv('data/main.tsv', sep='\t', low_memory=False)
df['date'] = pd.to_datetime(df['date'])


# Load df like in vae_train.py
isolate = Isolate(df[df['KeimName'] == 'Escherichia coli'].iloc[0], expand_mic=True)
ecoli = {k: v for k, v in zip(isolate.mic.abx, isolate.mic.e_log_profile)}
isolate = Isolate(df[df['KeimName'] == 'Escherichia coli (ESBL)'].iloc[0], expand_mic=True)
esbl = {k: v for k, v in zip(isolate.mic.abx, isolate.mic.e_log_profile)}


'''
On interpolation:

https://www.reddit.com/r/MLQuestions/comments/8l8u1e/how_to_interpolate_in_a_latent_space/

v_new = x v1 + (1-x) v2

And some theory:

https://towardsdatascience.com/what-a-disentangled-net-we-weave-representation-learning-in-vaes-pt-1-9e5dbc205bd1
'''

model.eval()

t1 = torch.Tensor(list(ecoli.values()))
t2 = torch.Tensor(list(esbl.values()))

v1 = get_latent([t1], model)[0]
v2 = get_latent([t2], model)[0]


mics = []
for i in np.arange(0, 1, 0.01)[::-1]:
    u = (i * v1) + ((1 - i) * v2)
    # This means we move from E. coli to P. mirabilis, and we should see
    # the colistin MIC increase.
    mic = model.decode(torch.Tensor(u))
    mics.append(mic)


foo = []
for i in mics:
    foo.append(np.exp(i[7].item()))
print(np.exp(foo[0]), np.exp(foo[-1]))
with open('interpolation.csv', 'w+') as out:
    out.write('step,MIC\n')
    for i, j in zip(np.arange(0, 1, 0.01), [np.exp(u) for u in foo]):
        out.write(f'{i},{j}\n')

