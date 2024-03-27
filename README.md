# EMNLP2023_classifiers
Code and data corresponding to **"Mandarin classifier systems optimize to accommodate communicative pressures"** - EMNLP conference 2023.

## Dataset
This study is based on three of the 1M sentence corpora of Mandarin Chinese in the [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download), and they are *2007-2009 news*, *2011 newscrawl*, and *2015 webcrawl*.

## Data extraction
* We normalized data by first transforming all chinese characters into simplified Chinese using the OPCC;
* We applied the CoreNLP Chinese dependency parser to our dataset;
* We extracted all complete norminal phrases from the dataset;
* We removed nominal phrases with unusual lengths (> 35 characters);
* We excluded nominal phrases with misclassified classifiers based on manual validation.

<img width="1049" alt="Screenshot 2024-03-27 at 12 15 44 PM" src="https://github.com/YameiW/EMNLP2023_classifiers/assets/49138382/0cddcfe9-fa85-421c-af5d-ec1c09eda8b0">


## Python files
There are six python files: 
* The script for data preprocessing to prepare the dataframes for running GAMs,
* The notebook for analyzing noun frequencies and MI in two structures,
* The script for calculating PMI (Pointwise Mutual Information) between pairs of nouns,
* The script for trainning customized vectors using both fasttext and word2vec models,
* The module of utils containing Functions used in the analysis,
* Results of manual validation.

## R files
The R Markdown file presents the outcomes of various GAMs.

## References

