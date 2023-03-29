import pandas as pd
from os import walk


# read in all COHA (Corpus of Historial American English) sample text files,
# concatenate into a giant string
corpus = ''

BASE = 'data/coha_sample/'
filenames = next(walk(BASE), (None, None, []))[2]  # [] if no file

for filename in filenames:
    with open(f'{BASE}{filename}') as f:
        text = f.read()
        corpus += ' ' + text

# make frequency dictionary
dic = pd.Series(corpus.split()).value_counts().to_frame().reset_index()
dic.columns = ['word', 'freq']

# at least 5 times
dic = dic.loc[dic['freq'] >= 5]

# exclude two or fewer chars
dic['nchar'] = dic['word'].str.len()
dic = dic.loc[dic['nchar'] >= 3]
dic.drop('nchar', axis=1, inplace=True)

# save
dic.to_csv('data/coha_dict.csv', index=False)
