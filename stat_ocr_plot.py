import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# aggretate stats by year
stat_ocr_sample = pd.read_csv('data/stat_ocr_sample.csv')
agg = stat_ocr_sample.groupby('year')[['num_words', 'num_errors', 'num_corrected']].mean()
agg.reset_index(inplace=True)
agg['year'] = agg['year'].astype('int')

# calculate percentages
agg['perc_error'] = agg['num_errors'] / agg['num_words']
agg['perc_corrected'] = agg['num_corrected'] / agg['num_errors']

# plot 1 - percentages
plt.figure(figsize=(8, 6))
plt.plot(agg['year'], agg['perc_corrected'], label='percent errors corrected')
plt.plot(agg['year'], agg['perc_error'], label='percent errors from words')

plt.legend()
plt.title('Percentage of non-word errors relative to all words\nand percentage of corrected relative to all errors')
plt.ylabel('percentage')
plt.xlabel('year')

plt.savefig('stat_ocr_plot_1.png')
# plt.show()

# plot 2 - raw values
plt.figure(figsize=(8, 6))
plt.plot(agg['year'], agg['num_words'], label='# words')
plt.plot(agg['year'], agg['num_errors'], label='# errors')
plt.plot(agg['year'], agg['num_corrected'], label='# corrected')

plt.legend()
plt.title('Raw counts of words, non-word errors & corrected errors per document')
plt.ylabel('count')
plt.xlabel('year')

plt.savefig('stat_ocr_plot_2.png')
