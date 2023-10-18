# this code was run on the server
import os
from joblib import Parallel, delayed
from gensim.models import Word2Vec
import fasttext
import sys
sys.path.append('../src')
from validation import *

def remove_clfs(filepath, clfs):
    '''
    This function removes clfs from the text and segments the text.
    '''
    with open(filepath, 'rt',encoding='utf-8') as file:
        data = file.readlines()

    ls = []
    a_str = ''
    for item in data:
        if item != '\n':
            word = item.split('\t')[1]
            if word not in clfs:
                a_str += word + ' '
        else:
            ls.append(a_str.strip())
            a_str = '' 
    return ls

def process_files_in_folder(folder_path, func, *args, n_jobs=-1):
    '''
    This function runs parallel processing of a function on all conllu files under a folder.
    '''
    file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.conllu')]
    results = Parallel(n_jobs=n_jobs)(delayed(func)(file_path, *args) for file_path in file_paths)
    return results

clfs = man_sortal+man_measure
folder_path ='/home/ywang78/scratch/conllu/leipzig_conllu'

# Process the files
results = process_files_in_folder(folder_path, remove_clfs, clfs)

# Combine results and write to a txt file
flattened_list = [sentence for sublist in results for sentence in sublist]
with open('corpora_chinese_noClfs.txt', 'w', encoding='utf-8') as out_file:
    for sentence in flattened_list:
        out_file.write(sentence + '\n')

# train a word2vec model
tokenized_sentences = [sent.split() for sent in flattened_list]

model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec_withoutClf.model")

# train a fasttext model without Clfs
flattened_list_noSpace = [item.replace(" ","") for item in flattened_list]

with open("./corpora_chinese_noClf_noSpace.txt", 'w', encoding='utf-8') as outfile:
    for sentence in flattened_list_noSpace:
        outfile.write(sentence+'\n')

model = fasttext.train_unsupervised('./corpora_chinese_noClf_noSpace.txt', minn=2, maxn=5, dim=300)       
model.save_model("./fasttext_withoutClf.bin")