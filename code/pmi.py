# this script is run on the cluster where the parsed conllu files are stored
import os
import pandas as pd
import pickle
from collections import defaultdict
import pyconll
from joblib import Parallel, delayed

def occurrence(filepath,noun_ls,window_size=2):
    '''
    this function get the bidirectional nouns around the target noun. Both the target noun and 
    the surrounding nouns are in the noun_ls 
    '''
    window_size = 2
    a_dict = defaultdict(list)
    data = pyconll.load_from_file(filepath)
    
    for sentence in data:
        for i, token in enumerate(sentence):
            if token.xpos == 'NN' and token.form in noun_ls:
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                phrase = [word.form for word in sentence[start:end] if (word.form in noun_ls) and (word.form != token.form)]
                
                for word in phrase:
                    if word not in a_dict[token.form]:
                        a_dict[token.form].append(word)
                        a_dict[word].append(token.form)
    return dict(a_dict)

def process_files_in_folder(folder_path, func, *args,n_jobs=-1):
    '''
    this function runs parallel processing of a function on all conllu files under a folder
    '''
    file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.conllu')]
    results = Parallel(n_jobs=n_jobs)(delayed(func)(file_path,*args) for file_path in file_paths)
    return results

pair_df1 = pd.read_csv("./clf_noun_structure.csv")
pair_df2 = pd.read_csv("./clf_mod_noun_structure.csv")

target_nouns1 = pd.concat([pair_df1['noun1'], pair_df1['noun2']]).unique()
target_nouns2 = pd.concat([pair_df2['noun1'], pair_df2['noun2']]).unique() 

folder_path = '/home/ywang78/scratch/conllu/leipzig_conllu'

occurrence_ls1 =  process_files_in_folder(folder_path, occurrence, target_nouns1)
occurrence_ls2 =  process_files_in_folder(folder_path, occurrence, target_nouns2)

with open('../data/clf_noun_pmi.pkl','wb') as file:
    pickle.dump(occurrence_ls1, file) 

with open('../data/clf_mod_noun_pmi.pkl','wb') as file:
    pickle.dump(occurrence_ls2, file)