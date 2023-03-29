# for each year, take a sample of 2 documents, compare their original and post-OCR corrected text
# calculate some stat

import pandas as pd
import pandas as pd
from editdistance import eval as distance
from datetime import datetime
from gc import collect


def correct(nonword_error):
    """
    A helper function at the correction step for one single non-word error.
    """
    global df_coha

    # calculate Damerau-Levenshtein (DL) distance between non-word error and each coha word
    df_coha['dist_with_curr_error'] = df_coha['word'].apply(lambda w: distance(nonword_error, w))

    # set distance thresold
    err_len = len(nonword_error)
    if err_len >= 3 and err_len <= 5:
        thresold = 1
    elif err_len >= 6 and err_len <= 9:
        thresold = 2
    else:
        thresold = 3

    # get correction candidates
    df_coha['is_candidate'] = df_coha['dist_with_curr_error'] <= thresold

    if df_coha['is_candidate'].sum() == 0:
        # uncorrectable, not processed to a correction, return original error word
        return nonword_error

    df_candidates = df_coha.loc[df_coha.is_candidate].reset_index(drop=True).copy()

    # correction step - find best candidate based on distance, then frequency
    for dist in range(1, thresold + 1):
        sub = df_candidates.loc[df_candidates['dist_with_curr_error'] == dist]
        if sub.shape[0] > 0:
            best = sub.nlargest(1, 'freq')
            best_correction = best.iloc[0, 0]
            return best_correction


def post_ocr_processing(ocr_text):
    """
    Processes one single piece of OCR text.
    ---
    ocr_text: string
    df_coha: specific format
    coha_set: specific format
    """
    global coha_set

    df_ocr_text = pd.DataFrame(ocr_text.split(), columns=['word'])

    # detect non-word errors in OCR text:
    # any word in the OCR text of length 3 or higher that cannot be found in the
    # reference dictionary of real words is flagged as a non-word error.
    df_ocr_text['nchar'] = df_ocr_text['word'].str.len()
    df_ocr_text['at_least_3'] = df_ocr_text['nchar'] >= 3
    df_ocr_text['not_in_set'] = df_ocr_text['word'].apply(lambda word: word not in coha_set)
    df_ocr_text['nonword_error'] = df_ocr_text['at_least_3'] & df_ocr_text['not_in_set']
    df_ocr_text.drop(['nchar', 'at_least_3', 'not_in_set'], axis=1, inplace=True)

    # apply correct() to the words that are nonword_errors
    df_ocr_text['corrected'] = df_ocr_text['word']
    df_ocr_text.loc[df_ocr_text['nonword_error'], 'corrected'] = df_ocr_text.loc[df_ocr_text['nonword_error'], 'word'].apply(correct)

    # some stat: num words, num non-word errors, num corrections
    return df_ocr_text.word.shape[0], df_ocr_text.nonword_error.sum(), (df_ocr_text['word'] != df_ocr_text['corrected']).sum()


if __name__ == '__main__':
    # get sample
    base_clean = '/n/henrich_lab/Lab/clean/'

    sub = pd.read_csv(f'{base_clean}strat_sample_200.csv')
    sub.drop_duplicates(inplace=True)
    collect()
    print(f'{datetime.now()} - subset taken; ready for ocr correction')
    print(f'{sub.shape[0]} documents')

    # COHA reference dictionary
    global df_coha
    global coha_set

    df_coha = pd.read_csv('data/coha_dict.csv')
    df_coha['word'] = df_coha['word'].astype('string')
    df_coha.dropna(inplace=True)

    coha_list = df_coha['word'].to_list()
    coha_set = set(coha_list)

    # post-OCR correction - correct all
    stat = pd.DataFrame(sub['text'].apply(post_ocr_processing).to_list(), columns=['num_words', 'num_errors', 'num_corrected'])
    print(f'{datetime.now()} - 100% correction done')

    # concat with id
    sub.drop('text', axis=1, inplace=True)
    sub.reset_index(drop=True, inplace=True)
    stat.reset_index(drop=True, inplace=True)
    res = pd.concat([sub, stat], axis=1)

    # save
    res.to_csv('stat_ocr_sample.csv', index=False)
    print(f'{datetime.now()} - result saved')
