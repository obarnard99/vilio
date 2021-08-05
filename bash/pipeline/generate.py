models = ['EL', 'ELV', 'ES', 'ESV']
feats = [1, 5, 10, 15, 20, 36, 50]
flags = ['', 'c']

out = ''
for model in models:
    for feat in feats:
        for flag in flags:
            out += '\'' + model + str(feat) + flag + '\' '
print('EXPERIMENTS=('+out+')')
