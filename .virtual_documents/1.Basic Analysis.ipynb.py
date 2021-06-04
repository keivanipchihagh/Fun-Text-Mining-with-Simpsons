import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


scripts_df = pd.read_csv('Data/Raw/simpsons_script_lines.csv', dtype = 'unicode')
episods_df = pd.read_csv('Data/Raw/simpsons_characters.csv', dtype = 'unicode')
characters_df = pd.read_csv('Data/Raw/simpsons_episodes.csv', dtype = 'unicode')
locations_df = pd.read_csv('Data/Raw/simpsons_locations.csv', dtype = 'unicode')


# Get character frequency
text_count_per_character = scripts_df['raw_character_text'].value_counts().sort_values(ascending = False)

# Draw bar plot for 10 most active characters
top = 10
plt.barh(text_count_per_character[:top].keys(), text_count_per_character[:top].values, label = 'Dialage Frequency')
plt.legend()
plt.show()


# Draw bar plot for 10 most active characters
top = 20
plt.barh(text_count_per_character[10:top].keys(), text_count_per_character[10:top].values, label = 'Dialage Frequency')
plt.legend()
plt.show()















