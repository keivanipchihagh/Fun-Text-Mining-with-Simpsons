import pandas as pd
import numpy as np
import re
import spacy

nlp = spacy.load('en_core_web_md')


scripts_df = pd.read_csv('Data/Raw/simpsons_script_lines.csv', dtype = 'unicode')
episods_df = pd.read_csv('Data/Raw/simpsons_episodes.csv', dtype = 'unicode')

scripts_df.head()


# episods_df
episods_df.drop(columns = ['image_url', 'original_air_year', 'video_url', 'production_code', 'views'], inplace = True)

# scripts_df
scripts_df.drop(columns = ['character_id', 'location_id', 'timestamp_in_ms', 'normalized_text', 'raw_text', 'number', 'id'], inplace = True)


# episodes_df
episodes_types = {
    'id': int,
    'imdb_rating': float,
    'imdb_votes': float,
    'number_in_season': int,
    'number_in_series': int,
    'season': int,
    'us_viewers_in_millions': float,    
}
episods_df = episods_df.astype(episodes_types)
episods_df['original_air_date'] = pd.to_datetime(episods_df['original_air_date'], format = 'get_ipython().run_line_magic("Y-%m-%d')", "")

# scripts_df
scripts_types = {
    'id': int,
    'episode_id': int,
    'character_id': int,
    'location_id': int,    
}
episods_df = episods_df.astype(episodes_types)


# Rename columns of episods_df
episods_df.drop(columns = ['id'], inplace = True)
episods_df.rename(columns = {'imdb_rating': 'imdb', 'original_air_date': 'datetime', 'us_viewers_in_millions': 'us_viewers'}, inplace = True)

# Convert 'us_viewers' from episods_df to original numeric format
episods_df['us_viewers'] = episods_df['us_viewers'].apply(lambda x: x * (10 ** 6))

# Drop unspokon scripts from scripts_df
scripts_df = scripts_df[scripts_df['speaking_line'] == 'true']
scripts_df.drop(columns = ['speaking_line'], inplace = True)
scripts_df.rename(columns = {'spoken_words': 'raw_text', 'raw_character_text': 'character', 'raw_location_text': 'location'}, inplace = True)


def normalize_text(text):
    return ' '.join([token.lemma_ for token in nlp(text.lower()) if (not token.is_stop and not token.is_punct)])    

scripts_df['normalized_text'] = scripts_df['raw_text'].apply(normalize_text)


scripts_df


scripts_df.to_csv('Data/Processed/scripts.csv', index = False)
episods_df.to_csv('Data/Processed/episodes.csv', index = False)
