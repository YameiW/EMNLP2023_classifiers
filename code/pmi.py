# this script is run on the cluster where the parsed conllu files are stored
import os
import pandas as pd
import pickle
from collections import defaultdict, Counter
import pyconll
from joblib import Parallel, delayed

def occurrence(filepath,noun_ls,window_size=2):
    '''
    this function get the bidirectional nouns around the target noun. Both the target noun and 
    the surrounding nouns are in the noun_ls 
    '''
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

def process_files_in_folder(folder_path, func, *args,n_jobs=-1,**kwargs):
    '''
    this function runs parallel processing of a function on all conllu files under a folder
    '''
    file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.conllu')]
    results = Parallel(n_jobs=n_jobs)(delayed(func)(file_path,*args,**kwargs) for file_path in file_paths)
    return results

def merge_occurrence_counter(occurrence):
    '''
    This function turns a list of dictionaries into a dictionary of counters: counters of occurrence of nouns
    '''
    merged = defaultdict(list)
    for d in occurrence:
        for key, value in d.items():
            merged[key].extend(value)
    for key in merged:
        merged[key] = Counter(merged[key])
    return merged

def process_and_merge(folder_path, occurrence, target_nouns, window_size=None):
    if window_size:
        occurrences = process_files_in_folder(folder_path, occurrence, target_nouns, window_size=window_size)
    else:
        occurrences = process_files_in_folder(folder_path, occurrence, target_nouns)
    
    return merge_occurrence_counter(occurrences)

pair_df1 = pd.read_csv("./clf_noun_structure.csv")
pair_df2 = pd.read_csv("./clf_mod_noun_structure.csv")

target_nouns1 = pd.concat([pair_df1['noun1'], pair_df1['noun2']]).unique()
target_nouns2 = pd.concat([pair_df2['noun1'], pair_df2['noun2']]).unique() 

folder_path = '/home/ywang78/scratch/conllu/leipzig_conllu'

# Define windows sizes and corresponding filenames
window_sizes = [None, 3, 5, 10]
file_names1 = [
    '../data/clf_noun_pmi.pkl',
    '../data/clf_noun_pmi_3win.pkl',
    '../data/clf_noun_pmi_5win.pkl',
    '../data/clf_noun_pmi_10win.pkl'
]
file_names2 = [
    '../data/clf_mod_noun_pmi.pkl',
    '../data/clf_mod_noun_pmi_3win.pkl',
    '../data/clf_mod_noun_pmi_5win.pkl',
    '../data/clf_mod_noun_pmi_10win.pkl'
]

for size, fname1, fname2 in zip(window_sizes, file_names1, file_names2):
    merged1 = process_and_merge(folder_path, occurrence, target_nouns1, size)
    merged2 = process_and_merge(folder_path, occurrence, target_nouns2, size)
    
    with open(fname1, 'wb') as file:
        pickle.dump(merged1, file)
    
    with open(fname2, 'wb') as file:
        pickle.dump(merged2, file)