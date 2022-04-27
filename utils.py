import json
import os
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob

import warnings
import pickle
from collections import defaultdict
from itertools import groupby
from scipy import stats

import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE


# functions for pooling

def get_phone_ali(ali_ctm, lang):
    phone_ali = pd.read_csv(ali_ctm, sep=" ", header=None)
    phone_ali.columns=["wav", "num", "onset", "dur_s", "phone_long", "None"]
    phone_ali["phone"] = phone_ali["phone_long"].apply(lambda x: x.split("_")[0])
    phone_ali["dur_ms"] = phone_ali["dur_s"].apply(lambda x: x*1000)
    phone_ali['wav'] = phone_ali['wav'].apply(lambda x: x.split('-')[1])


    if lang=="fr":
        phone_ali['phone'] = phone_ali['phone'].replace("au", "oo")
    return phone_ali

def sum_pool(row, w2feats):
    onset = math.ceil(row["onset"]*100)
    offset = math.ceil(row["onset"]*100+(row["dur_ms"]/10))
    cpc_feats_mat = w2feats[row["wav"]][onset:offset]
    return np.sum(cpc_feats_mat, axis=0)

def max_pool(row, w2feats):
    onset = math.ceil(row["onset"]*100)
    offset = math.ceil(row["onset"]*100+(row["dur_ms"]/10))
    cpc_feats_mat = w2feats[row["wav"]][onset:offset]
    return np.max(cpc_feats_mat, axis=0)

def avg_pool(row, w2feats):
    onset = math.ceil(row["onset"]*100)
    offset = math.ceil(row["onset"]*100+(row["dur_ms"]/10))
    cpc_feats_mat = w2feats[row["wav"]][onset:offset]
    return np.mean(cpc_feats_mat, axis=0)

def get_full_ali(w2feats, phone_ali, lang):
    phone_ali = phone_ali[phone_ali['phone']!="SIL"]
    phone_ali = phone_ali[phone_ali['wav'].isin(w2feats.keys())] 
    phone_ali['lang'] = lang
    phone_ali['max_pool'] = phone_ali.apply(max_pool, args=(w2feats,), axis = 1)
    phone_ali['sum_pool'] = phone_ali.apply(sum_pool, args=(w2feats,), axis = 1)
    phone_ali['avg_pool'] = phone_ali.apply(avg_pool, args=(w2feats,), axis = 1)
    return phone_ali


# functions to get alignment between phone and features
def get_w2feat(lang, root_path = '/gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/checkpoints/inftrain/EN+FR/3200h/00/features/cpc_small/'):
    
    
    if not os.path.isfile(os.path.join(root_path, 'CV_{}'.format(lang),'w2feat.pkl')):
        w2feat = {} 
        cv_txt = [x for x in os.listdir(os.path.join(root_path, 'CV_{}'.format(lang)))]

        for wav in tqdm(cv_txt):
            with open(os.path.join(root_path, 'CV_{}'.format(lang), wav)) as infile:
                try:
                    w2feat[wav.split('.')[0]] = np.loadtxt(infile)
                except:
                    print(wav)
        print("Saving to pickle")
        with open(os.path.join(root_path, 'CV_{}'.format(lang),'w2feat.pkl'), 'wb') as handle:
            pickle.dump(w2feat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(root_path, 'CV_{}'.format(lang),'w2feat.pkl'), 'rb') as handle:
            w2feat = pickle.load(handle)
    return w2feat



def retrieve_alignment(ali_ctm, lang, root_path = '/gpfsssd/scratch/rech/ank/ucv88ce/projects/MultilingualCPC/checkpoints/inftrain/EN+FR/3200h/00/features/cpc_small/'):
    
    if os.path.isfile(os.path.join(root_path, 'CV_{}'.format(lang),'full_alignment.pkl')):
        with open(os.path.join(root_path, 'CV_{}'.format(lang),'full_alignment.pkl'), 'rb') as handle:
            phone_ali = pickle.load(handle)
            
    else:

        print('1. Retrieveing w2feats')
        w2feat = get_w2feat(lang, root_path = root_path)

        print('2. Loading Alignment')
        phone_ali = get_phone_ali(ali_ctm, lang)

        print('3. Aligning CPC representations to phones')
        phone_ali = get_full_ali(w2feat, phone_ali, lang)
        with open(os.path.join(root_path, 'CV_{}'.format(lang),'full_alignment.pkl'), 'wb') as handle:
            pickle.dump(phone_ali, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return phone_ali
            

# phoneme lists
fr_phone_char = {'gn': 'nasal',
  'nn': 'nasal',
  'mm': 'nasal',
  'jj': 'fricative',
  'ss': 'fricative',
  'll': 'approximant',
  'bb': 'plosive',
  'kk': 'plosive',
  'vv': 'fricative',
  'zz': 'fricative',
  'gg': 'plosive',
  'ww': 'semi-vowel',
  'pp': 'plosive',
  'ff': 'fricative',
  'ch': 'fricative',
  'rr': 'fricative',
  'yy': 'approximant',
  'dd': 'plosive',
  'tt': 'plosive',
  'ou': 'vowel',
  'ei': 'vowel',
  'ii': 'vowel',
  'au': 'vowel',
  'aa': 'vowel',
  'ai': 'vowel',
  'on': 'nasal-vowel',
  'an': 'nasal-vowel',
  'oo': 'vowel',
  'oe': 'vowel',
  'eu': 'vowel',
  'ee': 'vowel',
  'un': 'nasal-vowel',
  'uu': 'vowel',
  'in': 'nasal-vowel',
  'uy': 'vowel'}

en_phone_char = {
    'dh': 'fricative', 'ah':'vowel','ah0':'vowel', 'v':'fricative', 'ae':'vowel', 'l':'approximant', 'iy':'vowel',
    'w':'approximant', 'z':'fricative', 'f':'fricative', 'ih':'vowel', 'd':'plosive', 'th':'fricative',
    'eh':'vowel', 'n':'nasal', 's':'fricative', 'ao':'vowel', 'g':'plosive', 'jh':'affricate', 'k':'plosive',
    'm':'nasal', 'ay':'vowel', 'er':'vowel', 'ow':'vowel', 'r':'approximant', 'y':'approximant', 'uw':'vowel', 
    'hh':'fricative', 't':'plosive', 'p':'plosive', 'sh':'fricative', 'uh':'vowel', 'aa':'vowel', 'ng':'nasal', 
    'ey':'vowel', 'b':'plosive', 'aw':'vowel', 'ch':'affricate', 'oy':'vowel', 'zh':'fricative'
}
