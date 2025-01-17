import os
import io
import glob
import json
import re
import pickle
from functools import reduce
from collections import defaultdict, Counter
from multiprocessing import Pool
from difflib import SequenceMatcher

import spacy
import numpy as np
from bs4 import BeautifulSoup
from spacy_langdetect import LanguageDetector
from termcolor import colored
from google.cloud import vision
from google.protobuf.json_format import MessageToJson
from PIL import Image
from loguru import logger
from tqdm import tqdm

# Word Lists
entity_blacklist = [
    'Stock photography',
    'Image',
    'Getty Images',
    'Royalty-free',
    'stock.xchng',
    'Photography',
    'ストックフォト',
    'Photo caption',
    'Clip art',
    'Illustration',
    'Quotation mark',
    'Portrait',
    'ʻOkina',
    'Apostrophe',
    'Quotation',
    'Punctuation',
    'Black and white',
    'Quotation marks in English',
    'Text',
    'Internet meme',
    'Stock illustration',
    'photo library',
    'Portable Network Graphics',
    'Photograph',
    'iStock',
    'Graphics',
    'photo caption',
    'Hawaiian language'
    'alamy',
    'illustration',
    'Shutterstock',
    'Poster',
    'Facebook',
    'Royalty payment',
    'E-book',
    'jpeg',
    'png',
    'Logo',
    'Vector graphics',
    'cartoon',
    'YouTube',
]
entity_blacklist = [e.lower() for e in entity_blacklist]
entity_whitelist = [
    'disable',
    'disability',
    'down syndrome',
    'downsyndrome',
    'immigran',
    'handicapped',
]
noun_chunk_blacklist = [
    'stock pictures',
    'stock picture',
    'stock photos',
    'stock photo',
    'premium high res pictures - getty image',
    'premium high res picture',
    'premium high res pictures - getty image',
    'photo and premium high res pictures - getty',
    'stock photo - download image',
    'high - res stock photo - getty image',
    '... - istock',
    '-pron-',
    'pictures',
    'photos',
    'premium high re',
    'royalty - free image',
    'royalty - free images - istock',
    'photo',
    'picture',
    'royalty - free photo & images - getty',
    'premium high',
    'royalty',
    'high resolution stock photography',
    'images - getty image',
    'royalty - free stock photo',
    'coronavirus',
    '- istock',
    '- photopin',
    'free image',
    'pinterest',
    'stock pictures',
    'pictures | getty image',
    'getty images',
    '- alamy',
    'royalty - free vector graphics',
    '- pinterest',
    'portrait photos',
    '- page',
    '- getty image',
    '- getty',
]
noun_chunk_blacklist = sorted(noun_chunk_blacklist, key=lambda x: len(x), reverse=True)

nlp = spacy.load("en_core_web_lg")


def create_img_list(img_dir, exclude_dir=None):
    """Returns list of pngs in img dir."""
    if exclude_dir is not None:
        eimg_list = glob.glob(os.path.join(exclude_dir, '*.png'))
        eimg_list += glob.glob(os.path.join(exclude_dir, '**', '*.png'))
        eimg_list = [os.path.basename(ei).split('.')[0] for ei in eimg_list]
    else:
        eimg_list = []

    img_list = glob.glob(os.path.join(img_dir, '*.png'))
    img_list += glob.glob(os.path.join(img_dir, '**', '*.png'))

    print(f"Found {len(img_list)} images")
    img_list = [im for im in img_list if os.path.basename(im).split('.')[0] not in eimg_list]
    print(f"Found {len(img_list)} images after filter")

    return img_list


def create_img_list_files(img_dir, output_dir='img_lists', split_size=30000, exclude_dir=None):
    """Creates text files of images to be processed."""
    img_list = create_img_list(img_dir, exclude_dir=exclude_dir)

    os.makedirs(output_dir, exist_ok=True)
    dir_name = os.path.basename(img_dir)

    for j, i in enumerate(range(0, len(img_list), split_size)):
        split = img_list[i: i + split_size]
        file_name = os.path.join(output_dir, f"{dir_name}_split.{j}.txt")
        with open(file_name, mode='w') as f:
            for l in split:
                f.write(l + '\n')


def detect_web(path):
    """Detects web annotations given path to an image."""

    assert os.path.exists(path) and os.path.isfile(path)

    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            print('\nBest guess label: {}'.format(label.label))

    if annotations.pages_with_matching_images:
        print('\n{} pages with matching images found'.format(len(annotations.pages_with_matching_images)))

    if response.error.message:
        raise Exception('{}\nFor more info on error messages, check: '
                        'https://cloud.google.com/apis/design/errors'.format(response.error.message))

    return annotations


def detect_image(path, json_path):
    """Calls web entity detection on a single image and stores result as json"""
    annotations = detect_web(path)
    json_str = MessageToJson(annotations)
    with open(json_path, mode='w') as f:
        f.write(json_str)


def detect_dataset(img_list, output_dir):
    """Generates json files with web entities for a list of images."""
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(img_list):
        print('-' * 100)
        assert os.path.exists(img)
        print(f"[{i}] {img}")

        img_name = os.path.basename(img)
        json_name = img_name.replace('.jpg', '').replace('.png', '') + '.json'
        json_path = os.path.join(output_dir, json_name)

        if os.path.exists(json_path):
            print(f'Skip {img}, it already exists!')
        else:
            detect_image(img, json_path)


def detect_dataset_from_file(img_list_file, output_dir):
    """Loads list of images and calls detect_dataset on it."""
    with open(img_list_file, mode='r') as f:
        img_list = f.readlines()
        detect_dataset(img_list, output_dir)


def create_description(json_dir, save=None):
    """Extract useful information from web entity jsons."""
    # Build list of json entity files
    json_list = glob.glob(os.path.join(json_dir, '*.json'))
    print(f'Found {len(json_list)} json files')
    assert len(json_list) > 0

    # Build json_map dict with img ids as keys and entity data as values
    json_map = defaultdict(lambda: {'main': None, 'split': {}})  # Returns this if key does not exist
    name_pat = re.compile(r'(\d+)\.?(\d)?\.json')
    for i, j in enumerate(json_list):
        file_name = os.path.basename(j)
        m = re.match(name_pat, file_name)
        assert m is not None, j
        try:
            with open(j, 'r') as f:
                content = json.load(f)
                if m.group(2) is None:
                    json_map[m.group(1)]['main'] = content
                else:
                    json_map[m.group(1)]['split'][int(m.group(2))] = content
        except Exception as e:
            print(i, j, e)
            continue

    search_math = [0, 0]
    entity_map = defaultdict(lambda: {})
    title_map = defaultdict(lambda: {})
    count_entity = Counter()
    num_entity = []
    for k, d in json_map.items():
        img_search = {0: d['main']} if len(d['split']) == 0 else d['split']
        for split_n, search in img_search.items():
            if 'pagesWithMatchingImages' in search:
                search_math[0] += 1

                # build entity list
                entity_name = [e['description'] for e in search['webEntities'] if 'description' in e]
                if 'label' in search['bestGuessLabels'][0]:
                    entity_name += [search['bestGuessLabels'][0]['label']]
                entity_name = [e for e in entity_name if e.lower() not in entity_blacklist]

                # build title list (with HTML tags removed)
                titles = [
                    BeautifulSoup(page['pageTitle'], 'html.parser').text
                    for page in search['pagesWithMatchingImages']
                    if 'pageTitle' in page
                ]

                entity_map[k][split_n] = entity_name
                title_map[k][split_n] = titles

                ent_count = Counter(entity_name)
                count_entity.update(ent_count)
                num_entity.append(len(entity_name))
            else:
                search_math[1] += 1

    num_entity = Counter(num_entity)

    if save is not None:
        with open(save, mode='wb') as pf:
            pickle.dump({
                'entity_map': dict(entity_map),
                'title_map': dict(title_map),
                'json_map': dict(json_map),
            }, pf)

    return dict(entity_map), dict(title_map), dict(json_map)


def clean_titles(title_map, save=None):
    snlp = spacy.load("en_core_web_lg")
    snlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
    print('Loaded SpaCy pipeline with language detector')

    titles = []
    ids = []
    split_idx = []
    for id, imgs_titles in title_map.items():
        for n, img_titles in imgs_titles.items():
            for t in img_titles:
                titles.append(t.lower())
                ids.append(id)
                split_idx.append(n)

    pipe = snlp.pipe(titles)

    # clean titles
    clean_title_map = defaultdict(lambda: defaultdict(list))  # TODO: change data structure?
    dropped = 0
    for i, doc in tqdm(enumerate(pipe)):
        # remove anything that isn't English
        if doc._.language['language'] != 'en':
            dropped += 1
            continue

        # reconstruct sentence, filtering blacklisted tokens
        senten = ' '.join([token.text.lower() for token in doc if token.pos_ not in ('NUM', 'X')])
        # for token in doc:
        #    if token.pos_ != 'NUM' and token.pos_ != 'X':
        #        senten += token.text.lower() + ' '
        for b in noun_chunk_blacklist:
            senten = re.sub(b, '', senten)
        for b in entity_blacklist:
            senten = re.sub(b, '', senten)

        clean_title_map[ids[i]][split_idx[i]].append(senten)

    # TODO: old code appended to pickle file here
    if save is not None:
        with open(save, mode='wb') as pf:
            pickle.dump(dict(clean_title_map), pf)

    return clean_title_map


def sent_cluster(roberta, embed_titles):
    embed = roberta.encode(embed_titles)

    for i, t in enumerate(embed_titles):
        print(f"[{i}]", t)

    norm = np.linalg.norm(embed, keepdims=True, axis=1)
    mtx_norm = np.matmul(norm, norm.T)
    mtx_cos = np.matmul(embed, embed.T) / mtx_norm

    cluster_mark = [-1] * len(embed_titles)
    cluster_cnt = 0
    for i, row in enumerate(mtx_cos):
        if cluster_mark[i] == -1:
            cluster_mark[i] = cluster_cnt
            cluster_cnt += 1
        for j in range(i, len(row)):
            if row[j] > 0.5 and cluster_mark[j] == -1:
                cluster_mark[j] = cluster_mark[i]
    print(cluster_mark)
    cus_size = Counter(filter(lambda x: x >= 0, cluster_mark))
    cluster_id, _ = cus_size.most_common(1)[0]
    print(cluster_id, _)

    gather_sent = [t for t, c in zip(embed_titles, cluster_mark) if c == cluster_id]
    return gather_sent


def link_noun_chunk(token, token_map, direction=None, depth=0, prev_token=None):
    if depth > 3:
        return []

    tk = token
    print(
        list(tk.children),
        colored(' --> ', color='blue'),
        tk,
        colored(' --> ', color='blue'),
        f"({tk.head}, {tk.head.pos_})",
        tk.dep_
    )
    token_link = [token]
    if direction is None:
        if tk.dep_ in ['compound', 'amod', 'poss', 'part']:
            if tk.head.pos_ in ['ADJ', 'NOUN', 'PROPN', 'PART']:
                token_link += link_noun_chunk(tk.head, token_map, direction='head', depth=depth + 1, prev_token=tk)
        if tk.children:
            for c in tk.children:
                if c.dep_ in ['poss', 'probj', 'amod', 'compound']:
                    token_link += link_noun_chunk(c, token_map, direction='child', depth=depth + 1, prev_token=tk)

    else:
        if tk.dep_ != 'ROOT' and direction == 'head':
            if tk.pos_ == 'ADP' and tk.dep_ == 'prep':
                token_link += link_noun_chunk(tk.head, token_map, direction='head', depth=depth + 1, prev_token=tk)
            elif tk.dep_ in ['compound', 'dep', 'amod', 'poss', 'part']:
                token_link += link_noun_chunk(tk.head, token_map, direction='head', depth=depth + 1, prev_token=tk)

        if tk.children:
            for c in tk.children:
                if c.dep_ in ['poss', 'compound'] and c != prev_token:
                    token_link += link_noun_chunk(c, token_map, direction='child', depth=depth + 1, prev_token=tk)
    return token_link


def extract_subject(titles):
    entity_cnt = defaultdict(lambda: 0)
    token_maps = []

    for title in titles:
        doc = nlp(title)
        entity2chunk = defaultdict(list)
        token_map = {}

        for token in doc:
            if len(token.text) == 1:
                continue
            if token.pos_ in ['NOUN', 'PROPN']:
                entity2chunk[token.text] += token.children
                entity_cnt[token.text] += 1
                token_map[token.text] = token
        print(dict(entity2chunk))
        token_maps.append(token_map)

    print(dict(entity_cnt))
    entity_cnt_list = sorted(list(entity_cnt.items()), key=lambda x: x[1], reverse=True)
    select_subject = [w for w, c in entity_cnt_list[:2]]

    result_noun_chunks = []
    subj_chunks = []

    for token_map in token_maps:

        for subj in select_subject:
            try:
                result = link_noun_chunk(token_map[subj], token_map)
                subj_chunks.append(result)
                print(colored("##", color='yellow'), result)
            except KeyError:
                pass
            print('-' * 100)

    if len(subj_chunks) > 1:
        cluster_mark = [-1] * len(subj_chunks)
        for i in range(0, len(subj_chunks)):
            if cluster_mark[i] == -1:
                cluster_mark[i] = max(cluster_mark) + 1
                for j in range(i + 1, len(subj_chunks)):
                    i_txt = {t.text for t in subj_chunks[i]}
                    j_txt = {t.text for t in subj_chunks[j]}
                    print(len(i_txt.union(j_txt)), len(j_txt) + len(i_txt))
                    if len(i_txt.union(j_txt)) < len(j_txt) + len(i_txt):
                        cluster_mark[j] = cluster_mark[i]
        unify_chunks = []
        for i in range(max(cluster_mark) + 1):
            cs = [sc for sc, j in zip(subj_chunks, cluster_mark) if j == i]
            # print('cs: ', cluster_mark)
            cs = reduce(lambda a, b: a + b, cs)
            unify_chunks.append(cs)
        result_noun_chunks += unify_chunks
    else:
        result_noun_chunks += subj_chunks

    # Gather result
    result_sent = []
    for title in titles:
        contain_white = any([w in title for w in entity_whitelist])
        if contain_white:
            result_sent.append({title})
    for tokens in result_noun_chunks:
        token_by_id = {}
        for t in tokens:
            token_by_id[t.i] = t
        result_sent.append({v.text for k, v in sorted(token_by_id.items(), key=lambda x: x[0])})
    return select_subject, result_sent


def get_best_match(query, corpus, step=4, flex=3, case_sensitive=False, verbose=False):
    """Return best matching substring of corpus.

    Parameters
    ----------
    query : str
    corpus : str
    step : int
        Step size of first match-value scan through corpus. Can be thought of
        as a sort of "scan resolution". Should not exceed length of query.
    flex : int
        Max. left/right substring position adjustment value. Should not
        exceed length of query / 2.

    Outputs
    -------
    output0 : str
        Best matching substring.
    output1 : float
        Match ratio of best matching substring. 1 is perfect match.
    """

    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + qlen - step <= len(corpus):
            match_values.append(_match(query, corpus[m: m - 1 + qlen]))
            if verbose:
                print(query, "-", corpus[m: m + qlen], _match(query, corpus[m: m + qlen]))
            m += step

        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(range(len(v)), key=v.__getitem__)

    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted
        # to optimize bmv_*
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + qlen] * 2

        # bmv_* are declared here in case they are untouched in optimization
        bmv_l = match_values[p_l // step]
        bmv_r = match_values[p_l // step]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

            if verbose:
                print("\n" + str(f))
                print("ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                print("lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                print("rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                print("rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))

        return bp_l, bp_r, _match(query, corpus[bp_l: bp_r])

    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)

    if flex >= qlen / 2:
        print("Warning: flex exceeds length of query / 2. Setting to default.")
        flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step

    pos_left, pos_right, match_value = adjust_left_right_positions()

    return slice(pos_left, pos_right), match_value


def remove_duplicate(titles):
    dup = [False for _ in range(len(titles))]
    for i, title in enumerate(titles[:-1]):
        if dup[i]:
            continue
        for j, title_b in enumerate(titles[i + 1:]):
            # print(title, title_b)
            if len(title) > len(title_b):
                match_slice, match_score = get_best_match(title_b, title, step=2, flex=4)
            else:
                match_slice, match_score = get_best_match(title, title_b, step=2, flex=4)
            if match_score > 0.85:
                dup[j + i + 1] = True
    filtered = [t for d, t in zip(dup, titles) if not d]
    return filtered


def apply_summary(args):
    id, n_split, titles = args
    print(colored(id, color='green'))
    titles = [re.sub('\|', '', e) for e in titles]
    titles = remove_duplicate(titles)
    select_subject, result_noun_chunks = extract_subject(titles)
    print('result_sent: ', result_noun_chunks)
    return id, n_split, result_noun_chunks


def summarise_titles(title_map, save=None):
    title_summaries = defaultdict(dict)
    with Pool(16) as pool:
        flatten = [
            (id, n_split, titles)
            for id, split_titles in title_map.items()
            for n_split, titles in split_titles.items()
        ]
        results = pool.map(apply_summary, flatten)
        for id, n_split, summary in results:
            title_summaries[id][n_split] = summary

    if save is not None:
        with open(save, mode='wb') as pf:
            pickle.dump(dict(title_summaries), pf)

    return title_summaries


def insert_anno_jsonl(title_summaries, entity_map, anno_json, split_boxes_json, ocr_boxes_json, img_dir):
    def refine_split_box(boxes, img_name):
        img_path = os.path.join(img_dir, img_name)
        w, h = Image.open(img_path).size

        if len(boxes) > 1:
            if w > h:
                # left-right
                boxes = sorted(boxes, key=lambda x: x[0])
                for j in range(len(boxes) - 1):
                    boxes[j + 1][0] = boxes[j][2] = (boxes[j][2] + boxes[j + 1][0]) / 2

                boxes[0][0] = 0
                boxes[-1][2] = w
                for j in range(len((boxes))):
                    boxes[j][1] = 0
                    boxes[j][3] = h
            else:
                # top-down
                boxes = sorted(boxes, key=lambda x: x[1])
                for j in range(len(boxes) - 1):
                    boxes[j][3] = boxes[j + 1][1] = (boxes[j][3] + boxes[j + 1][1]) / 2

                boxes[0][1] = 0
                boxes[-1][3] = h
                for j in range(len((boxes))):
                    boxes[j][0] = 0
                    boxes[j][2] = w
            return boxes
        else:
            return [[0, 0, w, h, 1.0]]

    def box_coverage(img_box, ocr_box):
        w = ocr_box[2] - ocr_box[0]
        h = ocr_box[3] - ocr_box[1]
        ocr_l_to_img_r = max(min(img_box[2] - ocr_box[0], w), 0)
        ocr_r_to_img_l = max(min(ocr_box[2] - img_box[0], w), 0)
        cover_w = min(ocr_l_to_img_r, ocr_r_to_img_l)

        ocr_t_to_img_b = max(min(img_box[3] - ocr_box[1], h), 0)
        ocr_b_to_img_t = max(min(ocr_box[3] - img_box[1], h), 0)
        cover_h = min(ocr_t_to_img_b, ocr_b_to_img_t)
        return (cover_h * cover_w) / (w * h)

    def merge_ocr_boxes(ocr_boxes):
        raise NotImplementedError

        box_id_pair = []
        font_height = []
        for i, box_info in enumerate(ocr_boxes):
            box, txt, score = box_info
            box_id_pair.append((i, box))
            font_height.append(box[3] - box[1])
        font_size = sum(font_height) / len(font_height)

        box_by_x = sorted(ocr_boxes, lambda box_info: box_info[0][0])
        box_by_xy = sorted(box_by_x, lambda box_info: box_info[0][1])
        box_by_y = sorted(ocr_boxes, lambda box_info: box_info[0][1])
        box_by_yx = sorted(box_by_y, lambda box_info: box_info[0][0])
        ln = len(box_by_xy)

        for i in range(ln - 1):
            box_by_xy[i]
            box_by_xy[i + 1]

    def find_appearance(meme_txt, ocr_boxes, img_boxes):
        # sort boxes
        box_by_x = sorted(ocr_boxes, key=lambda box_info: box_info[0][0])
        box_by_xy = sorted(box_by_x, key=lambda box_info: box_info[0][1])
        ocr_to_img_box = []

        # map text to image patch TODO: ?
        for i, box_info in enumerate(box_by_xy):
            box, txt, score = box_info
            imbox_cover = [box_coverage(im, box) for im in img_boxes]
            argmax = imbox_cover.index(max(imbox_cover))
            ocr_to_img_box.append(argmax)
        print('ocr_to_img_box: ', ocr_to_img_box)

        # match OCR text to original annotations
        for i, box_info in enumerate(box_by_xy):
            box, txt, score = box_info
            match_slice, match_score = get_best_match(txt, meme_txt, step=2, flex=4)
            print(
                colored(txt, color='blue'),
                colored(meme_txt[match_slice], color='green'),
                match_score,
                match_slice,
                len(meme_txt)
            )
            box_by_xy[i] = (box, txt, score, match_slice, match_score)

            if i > 0 and match_score >= 0.75:
                pass
                # assert match_slice.start >= box_by_xy[i - 1][3].stop
            elif match_score < 0.75:
                logger.warning(f"Low match score! {match_score}")

        char_to_img_box = [-1] * len(meme_txt)
        for i, box_info in enumerate(box_by_xy):
            match_slice, match_score = box_info[3:5]
            match_len = match_slice.stop - match_slice.start
            if match_len < 6 and match_score < 0.75:
                continue
            elif match_score < 0.6:
                continue

            for j in range(match_slice.start, min(match_slice.stop, len(meme_txt))):
                char_to_img_box[j] = ocr_to_img_box[i]

            if i < len(box_by_xy) - 1:
                next_slice = box_by_xy[i - 1][3]
                for j in range(match_slice.stop, next_slice.start):
                    char_to_img_box[j] = ocr_to_img_box[i]

        for i in range(len(char_to_img_box)):
            c = char_to_img_box[i]
            if c < 0:
                next_match = len(char_to_img_box)
                next_img_id = max(char_to_img_box)
                for j in range(i, len(char_to_img_box)):
                    if char_to_img_box[j] >= 0:
                        next_match = j
                        next_img_id = char_to_img_box[j]
                        break
                prev_img_id = max(char_to_img_box[:i]) if i > 0 else -1

                for j in range(i, next_match):
                    if prev_img_id == next_img_id:
                        char_to_img_box[j] = prev_img_id
                    elif prev_img_id < next_img_id:
                        char_to_img_box[j] = prev_img_id + 1
                    else:
                        # NOTE: happend when ocr's detected text not matching actual anno text.
                        if j > (i + next_match) // 2:
                            char_to_img_box[j] = next_img_id
                        else:
                            char_to_img_box[j] = prev_img_id

        print(char_to_img_box)
        print('-' * 100)
        return char_to_img_box

    with open(anno_json, 'r') as f:
        meme_anno = f.readlines()
        meme_anno = list(map(json.loads, meme_anno))

    with open(split_boxes_json, 'r') as f:
        image_split_annos = json.load(f)

    with open(ocr_boxes_json, 'r') as f:
        ocr_anno = json.load(f)

    for anno in meme_anno:
        id = anno['id']
        img_name = os.path.basename(anno['img'])

        image_boxes = image_split_annos[img_name]
        image_boxes = refine_split_box(image_boxes, img_name)
        ocr_boxes = ocr_anno[img_name]

        if len(image_boxes) > 1:
            logger.info(img_name)
            char_to_img_box = find_appearance(anno['text'], ocr_boxes, image_boxes)
        else:
            char_to_img_box = [0] * len(anno['text'])

        key = f"{id:05d}"
        if key in entity_map and key in title_summaries:
            img_entitys_splits = entity_map[key]
            summaries_splits = title_summaries[key]
        elif key in title_summaries:
            summaries_splits = title_summaries[key]
            img_entitys_splits = {k: [] for k in summaries_splits.keys()}
        elif key in entity_map:
            img_entitys_splits = entity_map[key]
            summaries_splits = {}
        else:
            img_entitys_splits = {}
            summaries_splits = {}

        anno['image_partition'] = []
        anno['partition_description'] = []
        anno['text_char_partition_id'] = char_to_img_box
        assert max(char_to_img_box) < len(image_boxes), f"{max(char_to_img_box)} !< {len(image_boxes)}"

        anno['image_partition'] = image_boxes
        for sn in range(len(image_boxes)):
            if sn in img_entitys_splits:
                entitys = img_entitys_splits[sn]
            else:
                entitys = []
            use_title = len(entitys) <= 2
            if use_title:
                if sn in summaries_splits:
                    web_page_summaries = summaries_splits[sn]
                    entitys += [' '.join(s) for s in web_page_summaries]
            anno['partition_description'].append(entitys)

    out_path = anno_json.replace('.json', '.entity.json')
    with open(out_path, 'w') as f:
        for anno_line in meme_anno:
            seri_line = json.dumps(anno_line)
            f.write(f"{seri_line}\n")


if __name__ == "__main__":
    # args
    feature_dir = 'C:/Users/obarn/Projects/F-MT126-1/vilio/data/features'
    img_dir = 'C:/Users/obarn/Projects/F-MT126-1/vilio/data/features/img_clean'
    split_img_dir = 'C:/Users/obarn/Projects/F-MT126-1/vilio/data/features/split_img_clean'
    entity_dir = 'C:/Users/obarn/Projects/F-MT126-1/vilio/data/features/entity_json'
    checkpoint_dir = 'C:/Users/obarn/Projects/F-MT126-1/vilio/data/features/checkpoints'
    anno_dir = 'C:/Users/obarn/Projects/F-MT126-1/vilio/data/features/annotations'

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "f-mt126-1-e8ab23b3ed9a.json"

    # build img list
    img_list = create_img_list(img_dir, exclude_dir=split_img_dir)
    img_list += create_img_list(split_img_dir)

    # mainloop
    # detect_dataset(img_list, entity_dir)  # Note: does not create separate split dir

    # parse
    if os.path.exists(os.path.join(checkpoint_dir, 'entity_tags.pickle')):
        print('Found entity_tags.pickle')
        with open(os.path.join(checkpoint_dir, 'entity_tags.pickle'), mode='rb') as pf:
            out = pickle.load(pf)
        entities, titles, all = out['entity_map'], out['title_map'], out['json_map']
    else:
        entities, titles, all = create_description(entity_dir, save=os.path.join(checkpoint_dir, 'entity_tags.pickle'))

    if os.path.exists(os.path.join(checkpoint_dir, 'clean_titles.pickle')):
        print('Found clean_titles.pickle')
        with open(os.path.join(checkpoint_dir, 'clean_titles.pickle'), mode='rb') as pf:
            titles = pickle.load(pf)
    else:
        titles = clean_titles(titles, save=os.path.join(checkpoint_dir, 'clean_titles.pickle'))

    if os.path.exists(os.path.join(checkpoint_dir, 'title_summaries.pickle')):
        print('Found title_summaries.pickle')
        with open(os.path.join(checkpoint_dir, 'title_summaries.pickle'), mode='rb') as pf:
            summaries = pickle.load(pf)
    else:
        summaries = summarise_titles(titles, save=os.path.join(checkpoint_dir, 'title_summaries.pickle'))

    # clean data

    # insert features
    for anno in ["train", "test_unseen", "test_seen", "dev_unseen", "dev_seen", "dev_all", "pretrain", "trainlarge",
                 "traindev"]:
        print(f'Inserting features into {anno}.jsonl')
        insert_anno_jsonl(summaries,
                          entities,
                          os.path.join(anno_dir, f'{anno}.jsonl'),
                          os.path.join(feature_dir, 'split_img_clean_boxes.json'),
                          os.path.join(feature_dir, 'ocr.box.json'),
                          img_dir)
