import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
from wordcloud import WordCloud

nlp = spacy.load('en_core_web_sm')


scripts_df = pd.read_csv('Data/Raw/simpsons_script_lines.csv', dtype = 'unicode')
episods_df = pd.read_csv('Data/Raw/simpsons_characters.csv', dtype = 'unicode')
characters_df = pd.read_csv('Data/Raw/simpsons_episodes.csv', dtype = 'unicode')
locations_df = pd.read_csv('Data/Raw/simpsons_locations.csv', dtype = 'unicode')

scripts_df.head()


# Get character frequency
text_count_per_character = scripts_df['raw_character_text'].value_counts().sort_values(ascending = False)

# Draw bar plot for 10 most active characters
top = 10
plt.barh(text_count_per_character[:top].keys()[::-1], text_count_per_character[:top].values[::-1])
plt.title('Ten Most Active Characters')
plt.xlabel('Dialoge Frequency')
plt.ylabel('Characters')
plt.show()


# Draw bar plot for 10 most active characters
top = 20
plt.barh(text_count_per_character[10:top].keys()[::-1], text_count_per_character[10:top].values[::-1])
plt.title('Next Ten Most Active Characters')
plt.xlabel('Dialoge Frequency')
plt.ylabel('Characters')
plt.show()


count_vectorizer = CountVectorizer(stop_words = 'english')
features = count_vectorizer.fit_transform(scripts_df[scripts_df['spoken_words'].isnull() == False]['spoken_words'])

# print(vectorizer.get_feature_names())

freqs = zip(count_vectorizer.get_feature_names(), features.sum(axis = 0).tolist()[0])
sorted_freqs = sorted(freqs, key = lambda x: -x[1])
print(sorted_freqs)


X, y = [], []

for freq in sorted_freqs[:20]:
    X.append(freq[0])
    y.append(freq[1])

fig = plt.figure(figsize = (6, 8))
plt.barh(X[::-1], y[::-1])
plt.title('Twenty Most Common Words')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()


wc = WordCloud(
    background_color = "white",
    colormap = "Dark2",
    max_font_size = 150,
)

all_text = []
for text in scripts_df['normalized_text']:
    if str(text) get_ipython().getoutput("= 'nan':")
        all_text.append(text)

wc.generate(' '.join(all_text))

plt.imshow(wc, interpolation = 'spline36')
plt.axis("off")
plt.show()


tfidf_vectorizer = TfidfVectorizer(stop_words = nlp.Defaults.stop_words)
features = tfidf_vectorizer.fit_transform(scripts_df[scripts_df['spoken_words'].isnull() == False]['spoken_words'])

# print(tfidf_vectorizer.get_feature_names())

freqs = zip(count_vectorizer.get_feature_names(), features.sum(axis = 0).tolist()[0])
sorted_freqs = sorted(freqs, key = lambda x: -x[1])
print(sorted_freqs)


X, y = [], []

for freq in sorted_freqs[:20]:
    X.append(freq[0])
    y.append(freq[1])

fig = plt.figure(figsize = (6, 8))
plt.barh(X[::-1], y[::-1])
plt.title('Twenty Most Common Words')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()






