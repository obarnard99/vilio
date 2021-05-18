models = ['U', 'O', 'D', 'X']
feats = [1, 5, 10, 15, 20, 36, 50, 72]
flags = ['', 'a', 'c', 'ac']

out = ''
for model in models:
    for feat in feats:
        for flag in flags:
            if 'c' in flag and feat == 72:
                continue
            out += '\'' + model + str(feat) + flag + '\' '
print('EXPERIMENTS=('+out+')')
