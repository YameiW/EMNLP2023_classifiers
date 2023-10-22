import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import fasttext
from gensim.models import Word2Vec
from itertools import combinations
from sklearn.utils import shuffle
from scipy.spatial import distance
import math
import pickle

def nounPair_df(noun_ls, a_dict):
    '''
    This function generates noun pairs based on sampled nouns(noun_ls), and the relative greaterness of n1 and n2 are balanced.
    a_dict is a dictionary of nouns and their frequencies.
    '''
    pair_ls = (list(combinations(noun_ls,2)))
    pair_df = pd.DataFrame(pair_ls,columns=['noun1','noun2'])

    pair_df['noun1_freq'] = pair_df['noun1'].map(a_dict)
    pair_df['noun2_freq'] = pair_df['noun2'].map(a_dict)

    pair_df = shuffle(pair_df).reset_index(drop=True)

    pair_df = balancedFreq_n1_n2(pair_df)

    return pair_df

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

def similarity_fasttext_custom(df,noun_ls):
    '''
    This function calculates the similarity between noun1 and noun2 based on the customed fasttext model
    '''
    ft = fasttext.load_model('/Users/yameiwang/Project/mason_project/EMNLP2023_classifiers/src/fasttext_withoutClf.bin')
    noun_vec = {} # get vectors for all nouns
    for noun in noun_ls:
        noun_vec[noun] = ft.get_word_vector(noun)

    def similarity(row):
        sim_score = 1.0 - distance.cosine(noun_vec[row['noun1']],noun_vec[row['noun2']])
        return sim_score
    
    df['sim_custom_fasttext'] = df.apply(similarity,axis=1)
    return df

def similarity_word2vec_custom(df,noun_ls):
    '''
    This function calculates the similarity between noun1 and noun2 based on the customed word2vec model.
    This function also generates the list of nouns that are not in the word2vec model, since some classifiers can be used nouns.
    '''
    ls = []
    model = Word2Vec.load("/Users/yameiwang/Project/mason_project/EMNLP2023_classifiers/src/word2vec_withoutClf.model")
    noun_vec = {} # get vectors for all nouns
    for noun in noun_ls:
        if noun in model.wv:
            noun_vec[noun] = model.wv[noun]
        else:
            ls.append(noun)
            noun_vec[noun] = np.zeros(100)

    def similarity(row):
        sim_score = 1.0 - distance.cosine(noun_vec[row['noun1']],noun_vec[row['noun2']])
        return sim_score
    
    df['sim_custom_word2vec'] = df.apply(similarity,axis=1)
    return df, ls

def entropy(labels):
    '''
    This function calculates the entropy of a list of labels
    '''
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    counts = np.array(list(Counter(labels).values()))
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    for i in probs:
        ent -= i * math.log(i, 2)
    return ent

def cond_entropy(noun_clf_counter):
    '''
    calculate the conditional entropy of nouns given a certain clf   
    based on the formular on Page 27 from Dye_2017.
    '''
    cond_n_exact_clf = {}
    for key1 in noun_clf_counter.keys():
        total_count = sum(noun_clf_counter[key1].values())
        result = 0.0
        for item in noun_clf_counter[key1]:
            prob = noun_clf_counter[key1][item]*1.0/total_count
            result += (-prob*math.log(prob,2))
        cond_n_exact_clf[key1] = result
    return cond_n_exact_clf


with open("../data/leipzig_noun.pkl",'rb') as file:
    nounFreq = pickle.load(file)

num_tokens = sum(nounFreq.values())

def pmi(n1,n2,occurrence, epsilon = 1e-8):
    '''
    This function calculates the pmi between noun pairs based on the formula
    pmi(x,y) = log(p(x,y)/p(x)p(y))
    num_tokens is the total frequencies of nouns in the dataset
    '''
    p_n1_n2 = (occurrence[n1][n2]+epsilon)/num_tokens
    p_n1 = (nounFreq[n1]+epsilon)/num_tokens
    p_n2 = (nounFreq[n2]+epsilon)/num_tokens
    div = p_n1_n2/(p_n1*p_n2)

    return math.log(div)

def pmi_apply(row,occurrence):
    try:
        pmi_score = pmi(row['noun1'],row['noun2'],occurrence)
        return pmi_score
    except TypeError: # where noun1 is in the occurrence, but with a empty list as key
        return 0
    
def process_occurrences(file_path, df1, df2, column_name):
    with open(file_path, 'rb') as file:
        occurrence = pickle.load(file)
    df1[column_name] = df1.apply(lambda row: pmi_apply(row, occurrence), axis=1)
    df2[column_name] = df2.apply(lambda row: pmi_apply(row, occurrence), axis=1)