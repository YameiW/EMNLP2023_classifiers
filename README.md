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

## Python files
There are three python files: 
* The notebook for data preprocessing to prepare the dataframes for running GAMs,
* The notebook for analyzing noun frequencies and MI in two structures,
* The script for calculating PMI (Pointwise Mutual Information) between pairs of nouns,
* The module of utils containing Functions used in the analysis,
* Results of manual validation.

## R files
The R Markdown file presents the outcomes of various GAMs.

## References

