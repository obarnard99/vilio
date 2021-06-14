models = ['U', 'O', 'D', 'X']
feats = [20]
flags = ['', 'a']

out = ''
for model in models:
    for feat in feats:
        for flag in flags:
            if 'c' in flag and feat == 72:
                continue
            out += '\'' + model + str(feat) + flag + '\' '
print('EXPERIMENTS=('+out+')')
