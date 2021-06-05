import pandas as pd
import numpy as np
import spacy

nlp = spacy.load('en_core_web_sm')


scripts_df = pd.read_csv('Data/Raw/simpsons_script_lines.csv', dtype = 'unicode')
characters_df = pd.read_csv('Data/Raw/simpsons_characters.csv', dtype = 'unicode')
episods_df = pd.read_csv('Data/Raw/simpsons_episodes.csv', dtype = 'unicode')
locations_df = pd.read_csv('Data/Raw/simpsons_locations.csv', dtype = 'unicode')


# lcoation_df
locations_df.drop(columns = ['name'], inplace = True)

# episods_df
episods_df.drop(columns = ['image_url', 'original_air_year', 'video_url', 'production_code', 'views'], inplace = True)

# characters_df
characters_df.drop(columns = ['normalized_name', 'gender'], inplace = True)

# scripts_df
scripts_df.drop(columns = ['word_count', 'raw_character_text', 'raw_location_text', 'timestamp_in_ms', 'normalized_text', 'raw_text', 'number', 'id'], inplace = True)


# location_df
locations_df['id'] = locations_df['id'].astype(int)

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

# characters_df
characters_df['id'] = characters_df['id'].astype(int)

# scripts_df
scripts_types = {
    'id': int,
    'episode_id': int,
    'character_id': int,
    'location_id': int,    
}
episods_df = episods_df.astype(episodes_types)


# Rename columns of episods_df
episods_df.rename(columns = {'imdb_rating': 'imdb', 'original_air_date': 'datetime', 'us_viewers_in_millions': 'us_viewers'}, inplace = True)

# Convert 'us_viewers' from episods_df to original numeric format
episods_df['us_viewers'] = episods_df['us_viewers'].apply(lambda x: x * (10 ** 6))

# Drop unspokon scripts from scripts_df
scripts_df = scripts_df[scripts_df['speaking_line'] == 'true']
scripts_df.drop(columns = ['speaking_line'], inplace = True)
scripts_df.rename(columns = {'spoken_words': 'raw_text'}, inplace = True)


scripts_df.to_csv('Data/Processed/simpsons_cleaned_script_lines.csv', index = False)
characters_df.to_csv('Data/Processed/simpsons_cleaned_characters.csv', index = False)
episods_df.to_csv('Data/Processed/simpsons_cleaned_episodes.csv', index = False)
locations_df.to_csv('Data/Processed/simpsons_cleaned_locations.csv', index = False)
