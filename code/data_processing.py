import pickle 
import pandas as pd
import numpy as np
import sys
sys.path.append('../src')
from utils import *

# Import cleaned data
df = pd.read_csv('../data/cleaned_data.csv',index_col=[0])
df = df.drop_duplicates()
df = df[df['clf_id'].astype('int')<df['clf_gov2_id'].astype('int')]
print(df.shape)
unique_num_noun = len(df['clf_gov2_form'].unique())
print(f"The number of unique nouns is {unique_num_noun}.")

# split the data into two structures: df1: clf_noun_structure; df2: clf_mod_noun_structure
df1 = df[df['clf_id']==df['clf_gov2_id']-1].reset_index()
df2 = df[df['clf_id'] != df['clf_gov2_id'] - 1].reset_index()

# data filter
# import the freq information
with open("../data/leipzig_noun.pkl",'rb') as file:
    nounFreq = pickle.load(file)

# create dataframe of modified nouns for two scenarios: clf_noun_structure and clf_mod_noun_structure
df1_nounFreq = pd.DataFrame(list(df1.clf_gov2_form.unique()),columns=['noun'])
df1_nounFreq['freq'] = df1_nounFreq.noun.map(nounFreq)
df2_nounFreq = pd.DataFrame(list(df2.clf_gov2_form.unique()),columns=['noun'])
df2_nounFreq['freq'] = df2_nounFreq.noun.map(nounFreq)

# remove nouns that are less than or equal to 25 in frequency
df1_nounFreq = df1_nounFreq[df1_nounFreq['freq']>25]
df2_nounFreq = df2_nounFreq[df2_nounFreq['freq']>25]

# sample nouns based on their frequency bins
df1_nounFreq['log_freq'] = np.log(df1_nounFreq['freq'])
df1_nounFreq['bins'] = pd.cut(df1_nounFreq['log_freq'],bins=30)
df1_sample = df1_nounFreq.groupby('bins').apply(lambda x: x.sample(frac = 0.01,replace=False, random_state=1)).reset_index(drop=True)

df2_nounFreq['log_freq'] = np.log(df2_nounFreq['freq'])
df2_nounFreq['bins'] = pd.cut(df2_nounFreq['log_freq'],bins=30)
df2_sample = df2_nounFreq.groupby('bins').apply(lambda x: x.sample(frac = 0.01,replace=False, random_state=1)).reset_index(drop=True)

# generate noun pairs
noun_ls1 = df1_sample['noun']
a_dict1 = {key:val for key,val in zip(df1_sample['noun'],df1_sample['freq'])}
pair_df1 = nounPair_df(noun_ls1,a_dict1)

noun_ls2 = df2_sample['noun']
a_dict2 = {key:val for key,val in zip(df2_sample['noun'],df2_sample['freq'])}
pair_df2 = nounPair_df(noun_ls2,a_dict2)

# similarities
pair_df1 = similarity(pair_df1,noun_ls1)
pair_df2 = similarity(pair_df2,noun_ls2)

pair_df1 = similarity_fasttext_custom(pair_df1,noun_ls1)
pair_df2 = similarity_fasttext_custom(pair_df2,noun_ls2)

pair_df1,ls1 = similarity_word2vec_custom(pair_df1,noun_ls1)
pair_df2,ls2 = similarity_word2vec_custom(pair_df2,noun_ls2)

with open('../data/word2vec_notfound_clfN.pkl', 'wb') as file:
    pickle.dump(ls1, file)

with open('../data/word2vec_notfound_clfModN.pkl', 'wb') as file:
    pickle.dump(ls2, file)

# pmi
file_names = [
    ('../data/clf_noun_pmi.pkl', '../data/clf_mod_noun_pmi.pkl', 'pmi'),
    ('../data/clf_noun_pmi_3win.pkl', '../data/clf_mod_noun_pmi_3win.pkl', 'pmi_win3'),
    ('../data/clf_noun_pmi_5win.pkl', '../data/clf_mod_noun_pmi_5win.pkl', 'pmi_win5'),
    ('../data/clf_noun_pmi_10win.pkl', '../data/clf_mod_noun_pmi_10win.pkl', 'pmi_win10')
]

for fname1, fname2, col_name in file_names:
    with open(fname1, 'rb') as file:
        occurrence1 = pickle.load(file)
    with open(fname2, 'rb') as file:
        occurrence2 = pickle.load(file)
    
    pair_df1[col_name] = pair_df1.apply(lambda row: pmi_apply(row, occurrence1), axis=1)
    pair_df2[col_name] = pair_df2.apply(lambda row: pmi_apply(row, occurrence2), axis=1)

# class membership
pair_df1 = class_mem_calculator(pair_df1,df,noun_ls1)
pair_df2 = class_mem_calculator(pair_df2,df,noun_ls2)

# save the data
pair_df1.to_csv('../data/clf_noun_structure.csv')
pair_df2.to_csv('../data/clf_mod_noun_structure.csv')