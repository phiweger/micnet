from pathlib import Path
import random

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.tseries.offsets import DateOffset
from tqdm import tqdm

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

from micnet.lab import Isolate
from micnet.utils import distance


def load_data(paths: list, complete_mic=True) -> DataFrame:
    print('Loading all chunks ...')
    chunks = []
    for fp in paths:
        chunks.append(pd.read_csv(fp, sep='\t', low_memory=False))

    print('Combining them ...')
    df = pd.concat(chunks)
    before = len(df)

    # Parse date but only keep day
    fmt = '%d.%m.%Y %H:%M'
    df['date'] = pd.to_datetime(df['AuftDatZeit'], format=fmt).apply(lambda x: x.date())
    
    df['ward'] = df.apply(lambda row: row['EinsCode'].split('_')[0], axis=1)
    # Z11_GCHS1 > Z11

    print('Parsing isolates, cleanup ...')
    IDs, ix, gram = [], [], []
    # hashes = []
    for _, i in tqdm(df.iterrows(), total=len(df)):
        surname, name, dob, *rest = i
        ID = f'{surname}::{name}::{dob}'
        IDs.append(ID)

        try:
            isolate = Isolate(i)
            ix.append(isolate.mic.complete)
            gram.append(isolate.stain)
        except ValueError:
            # "There are invalid numbers in the MIC profile"
            # These are records with more than one isolate of the same species
            ix.append(False)
            gram.append(float('nan'))
            continue

    assert len(IDs) == len(df)
    df['ID'] = IDs
    df['stain'] = gram

    # Remove all records with incomplete MIC (optional)
    if complete_mic:
        print(f'Remove records with incomplete MICs ...')
        df = df[ix]
    else:
        print(f'Records with incomplete MICs remain in the data!')

    # Remove swabs from "stuff" and the environment
    no_hygiene = []
    for _, i in df.iterrows():
        if ('#' in i['MatCode']) or ('Sterilkontrolle' in i['PatName']):
            no_hygiene.append(False)
        else:
            no_hygiene.append(True)
    df = df[no_hygiene]

    after = len(df)
    print(f'Raw data contains {before} rows, reduced to {after}')
    # 255,565 ->  250,541

    return df


class CustomImage(Dataset):
    '''
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    '''
    def __init__(self, filepath, state, profiles, model, transform=None):
        self.paths = list(
            (Path(filepath) / state).rglob('*.png'))
        self.profiles = profiles
        self.model = model
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def _get_latent(self, t, model):
        z = model.reparameterize(*model.encode(t))
        return z.detach().numpy()

    def __getitem__(self, ix):
        path = self.paths[ix]
        img = read_image(str(path))[0:3, :, :]
        # img = read_image(str(path))
        
        if self.transform:
            img = self.transform(img)
        
        label = path.parent.name

        x = np.array(random.choice(self.profiles[label]))
        t = torch.Tensor(x)

        self.model.eval()
        z = torch.Tensor(self._get_latent(t, self.model))

        return img, label, x, z


class CustomImage2(Dataset):
    '''
    To be used with the learning range finder, which expects fewer inputs.
    '''
    def __init__(self, filepath, state, profiles, model, transform=None):
        # path to images, lu table, model
        # load images
        # self.img_labels = pd.read_csv(annotations_file)
        self.paths = list(
            (Path(filepath) / state).rglob('*.png'))
        self.profiles = profiles
        self.model = model
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def _get_latent(self, t, model):
        z = model.reparameterize(*model.encode(t))
        return z.detach().numpy()

    def __getitem__(self, ix):
        path = self.paths[ix]
        img = read_image(str(path))[0:3, :, :]
        # img = read_image(str(path))
        
        if self.transform:
            img = self.transform(img)
        
        label = path.parent.name
        x = np.array(random.choice(self.profiles[label]))
        t = torch.Tensor(x)

        self.model.eval()
        z = torch.Tensor(self._get_latent(t, self.model))

        return img, z

