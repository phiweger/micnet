# .. continued from training


'''
> When I'm constructing a variational autoencoder, I like to inspect the latent dimensions for a few samples from the data to see the characteristics of the distribution. I encourage you to do the same. -- https://www.jeremyjordan.me/variational-autoencoders/
'''
data = [i[1] for i in zz]
plt.hist(data, bins=25, density=True, alpha=0.6, color='g'); plt.show()


# Grab MIC, embed, decode -- targets learned?
model.eval()
# u.mic.abx.index('CTX') 7
antibiotic = 'CTX'
invalid = 0
d = defaultdict(list)
for target in ['Escherichia coli (ESBL)', 'Escherichia coli']:
    for _, i in df[df['KeimName'] == target].iterrows():
        try:
            isolate = Isolate(i, expand_mic=True)
        except ValueError:
            invalid += 1
        t = torch.Tensor(isolate.mic.e_log_profile)
        z = get_latent([t], model)[0]
        recon = np.exp(model.decode(torch.Tensor(z)).detach().numpy())
        recon = [round(j, 4) for j in recon]
        d[target].append(recon[isolate.mic.abx.index(antibiotic)])


# Pseudomonas aeruginosa
model.eval()
# u.mic.abx.index('CTX') 7
invalid = 0
truth, yhat = [], []
for _, i in df[df['KeimName'] == 'Escherichia coli'].iterrows():
    try:
        isolate = Isolate(i, expand_mic=True)
    except ValueError:
        invalid += 1
    truth.append(isolate.mic.e_log_profile)
    t = torch.Tensor(isolate.mic.e_log_profile)
    z = get_latent([t], model)[0]
    recon = model.decode(torch.Tensor(z)).detach().numpy()
    recon = [round(j, 4) for j in recon]
    yhat.append(recon)


abx = isolate.mic.abx
with open('evaluate_profiles.csv', 'w+') as out:
    out.write('label,antibiotic,MIC\n')
    for i in truth:
        for k, v in zip(abx, i):
            out.write(f'truth,{k},{v}\n')
    for i in yhat:
        for k, v in zip(abx, i):
            out.write(f'yhat,{k},{v}\n')
