from collections import Counter, defaultdict
import copy
from glob import glob
import json
import os
import random
import time

import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, models, transforms
device = torch.device('cuda' if 0 else 'cpu')

from micnet.data import CustomImage
from micnet.lab import Isolate
from micnet.models import VAE, get_latent, decode


# README: Access to the raw "bk" data is restricted.
# Load data from positive blood cultures
# from observatory.data import load_data
# fp = data/bk.tsv'
# df = load_data([fp], complete_mic=True)  # 58,075 records
# Raw data contains 58075 rows, reduced to 17866
# out = 'data/bk.cleaned.tsv'
# df.to_csv(out, index=False, sep='\t')
out = 'data/bk.cleaned.tsv'
df = pd.read_csv(out, sep='\t', low_memory=False)
df['date'] = pd.to_datetime(df['date'])

# Deduplicate
df = df.sort_values('date').drop_duplicates('ID', keep='last')
d = defaultdict(list)
cnt = 0
for _, i in df.iterrows():
    try:
        isolate = Isolate(i, expand_mic=True)
    except ValueError:
        cnt += 1
    d[isolate.species].append(isolate.mic.e_log_profile)
# 10451 entries, 2 failed
selection = set(['Escherichia coli', 'Staphylococcus aureus', 'Pseudomonas aeruginosa', 'Streptococcus agalactiae', 'Enterococcus faecium', 'Enterococcus faecalis', 'Proteus mirabilis'])
d_ = {k.replace(' ', '_').lower(): v for k, v in d.items() if k in selection}


# Load VAE model
fp = 'data/vae.pt'
dim = 26  # expanded MIC
model = VAE(dim).to(device)
state = torch.load(fp)
model.load_state_dict(state)
# <All keys matched successfully>


# Load data for image model
input_size = 224
batch_size = 64

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # normalize to ImageNet images
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Dataloader
fp = 'data/crops'  # https://osf.io/n3247/
image_datasets = {state: CustomImage(fp, state, d_, model, data_transforms[state]) for state in ['train', 'val', 'test']}

dataloaders_dict = {state: torch.utils.data.DataLoader(image_datasets[state], batch_size=batch_size, shuffle=True) for state in ['train', 'val', 'test']}


'''
Aim: Finetune a DL vision model for bacterial image classification, and ultimately mapping images into the MIC VAE space for a personalized calculated therapy.

https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
'''
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_resnet(num_classes, feature_extract, use_pretrained=True):
    '''
    TODO: Larger ResNet?

    # https://pytorch.org/hub/pytorch_vision_resnet/
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    '''
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()

    # val_acc_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
    best_loss = 1e6

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, species, profile, embedding in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = embedding.to(device)

                # Train ResNet on image labels
                # labels = torch.Tensor([1 if i == 'escherichia_coli' else 0 for i in species]).type(torch.LongTensor)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # print(outputs[:5])
                    # print(labels[:5])

                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #    best_acc = epoch_acc
            #    best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_loss < best_loss:
               best_loss = epoch_loss
               best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #    val_acc_history.append(epoch_acc)
            if phase == 'val':
               val_loss_history.append(epoch_loss)

    return model, val_loss_history


num_classes = 2
num_epochs = 10
feature_extract = True


# Initialize the model for this run
model_ft, input_size = initialize_resnet(
    num_classes, feature_extract, use_pretrained=True)

# Send the model to GPU
model_ft = model_ft.to(device)


# Gather the parameters to be optimized/updated in this run. If we are
# finetuning we will be updating all parameters. However, if we are
# doing feature extract method, we will only update the parameters
# that we have just initialized, i.e. the parameters with requires_grad
# is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
'''
Params to learn:
     fc.weight
     fc.bias
'''


'''
# LR finder:
# https://github.com/davidtvs/pytorch-lr-finder
# Multi-output regression difficult:
# https://github.com/davidtvs/pytorch-lr-finder/issues/35
# !pip install torch-lr-finder
from torch_lr_finder import LRFinder
fp = 'data/crops'
image_datasets = {state: CustomImage2(fp, state, d_, model, data_transforms[state]) for state in ['train', 'val', 'test']}
dataloaders_dict = {state: torch.utils.data.DataLoader(image_datasets[state], batch_size=batch_size, shuffle=True) for state in ['train', 'val', 'test']}

lr_finder = LRFinder(model_ft, optimizer_ft, criterion, device=device)
lr_finder.range_test(dataloaders_dict['train'], start_lr=3e-4, end_lr=10, num_iter=100)
# use CustomImage__2__ loader
# lr_finder.plot()
# LR suggestion: steepest gradient, Suggested LR: 1.18E-03
'''


# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer_ft, max_lr=0.001, steps_per_epoch=len(dataloaders_dict['train']), epochs=num_epochs)


# Setup the loss fxn
# For images:
# criterion = nn.CrossEntropyLoss()
# For vectors:
criterion = nn.MSELoss()

# Train and evaluate
model_ft, hist = train_model(
    model_ft,
    dataloaders_dict,
    criterion,
    optimizer_ft,
    scheduler=None,
    num_epochs=num_epochs)


abbrev = {
    'escherichia_coli': 'E. coli',
    'pseudomonas_aeruginosa': 'P. aeruginosa',
    'enterococcus_faecalis': 'Ent. faecalis',
    'staphylococcus_aureus': 'S. aureus',
}



fp = 'data/eucast_breakpoints.json'
with open(fp, 'r') as file:
    bp = json.load(file)


model_ft.eval()
model.eval()
performance, diffs = [], []
abx = ['AMK', 'AMS', 'AZT', 'CAZ', 'CIP', 'CLI', 'COL', 'CTX', 'CXM', 'DOX', 'DPT', 'FOS', 'GEN', 'IMP', 'LEV', 'LIZ', 'MER', 'MOX', 'PIP', 'PIT', 'RAM', 'ROX', 'SXT', 'TOB', 'TPL', 'VAN']


with open('pred.csv', 'w+') as out:
    out.write('species,c1,c2,label\n')

    for img, species, mics, latent in tqdm(iter(dataloaders_dict['test'])):
        pred = model_ft(img)
        # model_ft(foo[0].unsqueeze(0))
        yhat = decode(pred, model)
        species = [abbrev[s] for s in species]

        # Performance
        for ix in range(len(species)):
            tp, fp, tn, fn = 0, 0, 0, 0
            
            a = np.exp(mics.detach().numpy()[ix])
            b = np.exp(yhat[ix])
            
            for substance, diff in zip(abx, np.round(a - b, 4)):
                if bp[species[ix]][substance]:
                    diffs.append([species[ix], substance, diff])

            for i, j, p in zip(a, b, bp[species[ix]].values()):
                
                if not p:
                    continue
                if i <= p and j <= p:
                    tp += 1
                elif i <= p and j > p:
                    fn += 1
                elif i > p and j <= p:
                    fp += 1
                else:
                    tn += 1

            try:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                acc = (tp + tn) / (tp + fp + tn + fn)
                fscore = (2 * precision * recall) / (precision + recall)
                # round(pos / (pos + neg), 4)
                performance.append(
                    [species[ix], precision, recall, acc, fscore])
            except ZeroDivisionError:
                pass


        for (c1, c2), name in zip(pred, species):
            out.write(f'{name},{c1},{c2},prediction\n')
        for (c1, c2), name in zip(latent, species):
            out.write(f'{name},{c1},{c2},truth\n')


df_performance = pd.DataFrame(performance, columns=['species', 'precision', 'recall', 'acc', 'fscore'])
df_performance.to_csv('performance.csv', index=False)

df_diffs = pd.DataFrame(diffs, columns=['species', 'substance', 'delta'])
df_diffs.to_csv('diffs.csv', index=False)
