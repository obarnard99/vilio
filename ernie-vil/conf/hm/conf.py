
import json

exps = ['1', '1a', '1c', '1ac', '5', '5a', '5c', '5ac', '10', '10a', '10c', '10ac', '15', '15a', '15c', '15ac', '20', '20a', '20c', '20ac', '36', '36a', '36c', '36ac', '50', '50a', '50c', '50ac']

for feats in exps:
    with open('task_hm.json') as f:
        task_group = json.load(f)
        task_group[0]['feature_lmdb_path'] = f'./data/hm/features/tsv/{feats}.tsv'
        if 'ac' in feats:
            flags = 'ac'
        elif 'a' in feats:
            flags = 'a'
        elif 'c' in feats:
            flags = 'c'
        else:
            flags = ''  
        task_group[0]['gt_feature_lmdb_path'] = f'./data/hm/features/tsv/10100{flags}.tsv'
        task_group[0]['use_gt_fea'] = True
    with open(f'task_hm_{feats}.json', 'w+') as f:
        json.dump(task_group, f)
