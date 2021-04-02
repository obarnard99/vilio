# Collection of pandas scripts that may be useful
import pandas as pd
import os

from PIL import Image
import imagehash


# Image hash functions: 
# https://content-blockchain.org/research/testing-different-image-hash-functions/

def phash(img_path):
    # Identifies dups even when caption is different
    phash = imagehash.phash(Image.open(img_path))
    return phash


# Data Cleaning

# The HM Dataset is very noisy: In the first version of the dataset there were many duplicates with conflicting labels
# In the second version, the conflicting labels have all been resolved, yet the duplicates remain
def clean_data(data_path="./data", img_path="./data", force=False):
    """
    Cleans the HM train & dev data.
    Outputs traindev & pretrain data.

    data_path: Path to folder with train.jsonl, dev_unseen.jsonl, dev_seen.jsonl
    """
    # Check if the statement was already run and the necessary data exists:
    if os.path.exists(os.path.join(data_path, "pretrain.jsonl")):
        print('Clean datasets already exist')
        if not force:
            return
        print('Rebuilding clean datasets...')
    else:
        print("Preparing...")

    # Load all files
    train = pd.read_json(os.path.join(data_path, "train.jsonl"), lines=True, orient="records")
    dev_seen = pd.read_json(os.path.join(data_path, "dev_seen.jsonl"), lines=True, orient="records")

    # We validate with dev_seen throughout all experiments, so we only take the new data from dev_unseen add it to
    # train and then discard dev_unseen
    dev_unseen = pd.read_json(os.path.join(data_path, "dev_unseen.jsonl"), lines=True, orient="records")
    dev_unseen = dev_unseen[~dev_unseen['id'].isin(dev_seen.id.values)].copy()

    # Clean training data
    df_dict = {'train': train, 'dev_seen': dev_seen, 'dev_unseen': dev_unseen}
    train_dist = pd.concat([df.assign(identity=key) for key, df in df_dict.items()])
    train_dist['full_path'] = train_dist['img'].apply(lambda x: os.path.join(img_path, str(x)))

    # Identify text dups
    text_dups = train_dist.text.value_counts().reset_index(name="counter")
    text_dups = text_dups.loc[text_dups['counter'] > 1]

    rmv_ids = []
    for t in text_dups['index'].values:
        # Identify image dups
        text_dup_df = train_dist.loc[train_dist.text == t].copy()
        text_dup_df['hash'] = text_dup_df['full_path'].apply(lambda x: phash(x))
        hash_dups = text_dup_df.hash.value_counts().reset_index(name="counter")
        hash_dups = hash_dups.loc[hash_dups['counter'] > 1]

        for h in hash_dups['index'].values:
            # Identify correct label by majority rule
            dup_df = text_dup_df.loc[text_dup_df.hash == h]
            true_label = round(dup_df.label.values.mean())

            # Add duplicate IDs to rmv_ids except for last one
            rmv_ids.extend(dup_df.loc[dup_df.label != true_label].id.values)
            rmv_ids.extend(dup_df.loc[dup_df.label == true_label].id.values)
            rmv_ids.pop()

    # Output all files we need

    # a) Clean train file (All duplicates are in train)
    train = train[~train['id'].isin(rmv_ids)].copy()
    train.to_json(path_or_buf=os.path.join(data_path, "train.jsonl"), orient='records', lines=True)

    # b) Pretrain file for ITM & LM pre-training
    pretrain = pd.concat([train, dev_seen, dev_unseen])
    pretrain.to_json(path_or_buf=os.path.join(data_path, "pretrain.jsonl"), orient='records', lines=True)

    # c) Cleaned Train + unused data from dev_unseen
    trainlarge = pd.concat([train, dev_unseen])
    trainlarge.to_json(path_or_buf=os.path.join(data_path, "trainlarge.jsonl"), orient='records', lines=True)

    # d) Cleaned Train + unused data from dev_unseen + dev_seen
    traindev = pd.concat([train, dev_seen, dev_unseen])
    traindev.to_json(path_or_buf=os.path.join(data_path, "traindev.jsonl"), orient='records', lines=True)

    # e) Full dev set
    dev_all = pd.concat([dev_seen, dev_unseen])
    dev_all.to_json(path_or_buf=os.path.join(data_path, "dev_all.jsonl"), orient='records', lines=True)


if __name__ == '__main__':
    clean_data(data_path=r'C:\Users\obarn\Projects\F-MT126-1\vilio\data\features\annotations', img_path=r'C:\Users\obarn\Projects\F-MT126-1\data\hmc_unseen', force=True)
