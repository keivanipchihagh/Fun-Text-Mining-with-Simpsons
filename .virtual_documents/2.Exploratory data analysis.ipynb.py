import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import spacy

nlp = spacy.load('en_core_web_md')


scripts_df = pd.read_csv('Data/Processed/scripts.csv')
episods_df = pd.read_csv('Data/Processed/episodes.csv')

scripts_df.head(3)


top_ten_active_characters = scripts_df.groupby(
    by = 'character',
    as_index = False
).count().sort_values(
    by = 'raw_text',
    ascending = False
).iloc[:10]

plot = top_ten_active_characters[['character', 'raw_text']].plot.bar(x = 'character', y = 'raw_text', label = 'Dialogs', rot = 30)
fig = plot.get_figure()
fig.set_size_inches(20, 5)
fig.savefig("Plots/top_ten_active_characters.png")
plt.show()


normalized_text_values = scripts_df.loc[scripts_df['normalized_text'].notnull(), 'normalized_text'].values
twenty_top_common_words = pd.Index(' '.join(normalized_text_values).split(' ')).value_counts()[:20]

fig = plt.figure(figsize = (20, 5))
plt.bar(x = twenty_top_common_words.index, height = twenty_top_common_words.values, label = 'Repetition')
plt.legend()
plt.savefig('Plots/twenty_top_common_words.png')
plt.show()



























