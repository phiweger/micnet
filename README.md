## README

`MICNet` is a chimeric neural net that is used to predict antimicrobial susceptibility profiles from microscopic images. It uses an idea from the `DALL-E` net from OpenAI: We first train a variational autoencoder (VAE) on a bunch of susceptibility data (unsupervised). "Susceptibility" here is defined as _minimum inhibitory concentration_ (MIC) measurements across a panel of antibiotics using the dilution broth method. Then, we chop off the encoder and replace it with a (pretrained) model, here an image model (`ResNet`). We finetune it to map image representations into the latent space of the VAE and then use the VAE decoder to generate MIC profiles for a given input image.


### Install

```bash
conda create -n micnet python=3.8 -c conda-forge numpy pandas tqdm matplotlib

# Install your system-specific version of PyTorch, e.g. for Mac:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio -c pytorch

# Install some utils
git clone www.github.com/phiweger/micnet
cd micnet && pip install -e .

# If you want to experiment with different learning rate:
# https://github.com/davidtvs/pytorch-lr-finder
pip install torch-lr-finder
```


### Run

_Note that the original data is under restricted access due to university regulations. It is provided, however, upon reasonable request, and subject to terms and conditions. Below, we nevertheless provide the code used to construct our predictive model(s)._

To predict susceptibility profiles from images we use two datasets:

1. A pretrained VAE in `data/vae.pt`. To find out how we trained and evaluated the model, see:

- `scipts/vae_train.py`
- `scipts/vae_interpolate.py`
- `scipts/vae_evaluate.py`

2. Example images from a previously published dataset (\*); however, we chopped them up into `crops` available from OSF (https://osf.io/n3247/). You find a copy of the original data there, too. Cropping was done as specified in `scripts/prepare_dibs_data.py`. 

Finally, to do the actual predicting, we used `scripts/resnet_predict.py`. Note that it will download a `ResNet` with pretrained weights.

In the `data` folder, you'll also find intermediate results.

---

(\*): Zieliński, Bartosz, Anna Plichta, Krzysztof Misztal, Przemysław Spurek, Monika Brzychczy-Włoch, and Dorota Ochońska. 2017. “Deep Learning Approach to Bacterial Colony Classification.” PloS One 12 (9): e0184554.
