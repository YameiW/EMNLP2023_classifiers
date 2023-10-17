import pandas as pd
import numpy as np
from collections import defaultdict
import fasttext
from scipy.spatial import distance



def balancedFreq_n1_n2(df):
    '''
    This function would generate the dataframe with the balanced number of noun1_freq and noun2_freq.
    Therefore, there are equvilant number of noun1_freq > noun2_freq and noun1_freq < noun2_freq.
    '''
    total_num = df.shape[0]
    f1 = df[df['noun1_freq']>df['noun2_freq']].shape[0]
    f2 = df[df['noun1_freq']<df['noun2_freq']].shape[0]
    f3 = df[df['noun1_freq']==df['noun2_freq']].shape[0]
    num = int(max(f1,f2)-(total_num - f3)/2)

    idx0 = df[df['noun1_freq']>df['noun2_freq']].index

    idx1 = df[df['noun1_freq']< df['noun2_freq']].sample(n=num, replace=False, random_state=2023).index
    idx2 = np.setdiff1d(df[df['noun1_freq']<df['noun2_freq']].index.to_numpy(),idx1.values)
    idx3 = df[df['noun1_freq']==df['noun2_freq']].index

    df_a = df.iloc[idx0]
    df_b = df.iloc[idx1]\
                .rename(columns={'noun1':'noun2','noun2':'noun1','noun1_freq':'noun2_freq','noun2_freq':'noun1_freq',
                            'noun1_log':'noun2_log','noun2_log':'noun1_log'})
    df_c = df.iloc[idx2]
    df_d = df.iloc[idx3]

    df = pd.concat([df_a,df_b,df_c,df_d]).reset_index(drop=True)

    return df

def class_mem_calculator(df1, df2, noun_ls):
    '''
    The function calculates the summed proportion of the two classes in noun pairs.
    df1: the dataframe contains noun pairs
    df2: the dataframe contains the association information of nouns and classifiers
    noun_ls: the list of sampled nouns
    '''
    def n_clf(row,noun):
        return len(noun_clf[row[noun]])
    
    def shared_clf(row):
        return len(noun_clf[row['noun1']].intersection(noun_clf[row['noun2']]))

    noun_clf = defaultdict(set)
    for key,val in zip(df2['clf_gov2_form'],df2['clf_form']):
        noun_clf[key].add(val)
    noun_clf = {key:val for key,val in noun_clf.items() if key in list(noun_ls)}

    df1['n1_clf_cnt'] = df1.apply(lambda row: n_clf(row,'noun1'),axis=1)
    df1['n2_clf_cnt'] = df1.apply(lambda row: n_clf(row,'noun2'),axis=1)
    df1['shared_clf_cnt'] = df1.apply(shared_clf, axis=1)

    df1['sum_prop'] = (df1['shared_clf_cnt']/df1['n1_clf_cnt'])+(df1['shared_clf_cnt']/df1['n2_clf_cnt'])
    return df1

def similarity(df,noun_ls):
    '''
    This function calculates the similarity between noun1 and noun2 based on the fasttext model
    '''
    ft = fasttext.load_model('/Users/yameiwang/Project/mason_project/EMNLP2023_classifiers/src/cc.zh.300.bin')
    noun_vec = {} # get vectors for all nouns
    for noun in noun_ls:
        noun_vec[noun] = ft.get_word_vector(noun)

    def similarity(row):
        sim_score = 1.0 - distance.cosine(noun_vec[row['noun1']],noun_vec[row['noun2']])
        return sim_score
    
    df['sim'] = df.apply(similarity,axis=1)

    return df