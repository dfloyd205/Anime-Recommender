import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

anime = pd.read_csv("anime_cleaned.csv")

genres = anime["genre"].str.lower()

categories = ['action', 'adventure', 'cars', 'comedy', 'dementia', 'demons', 'drama', 'ecchi', 'fantasy', 'game', 'harem', 'hentai', 'historical', 'horror', 'josei', 'kids', 'magic', 'martial arts', 'mecha', 'military', 'music', 'mystery', 'parody', 'police', 'psychological', 'romance', 'samurai', 'school', 'sci-fi', 'seinen', 'shoujo', 'shoujo ai', 'shounen', 'shounen ai', 'slice of life', 'space', 'sports', 'super power', 'supernatural', 'thriller', 'vampire', 'yaoi', 'yuri']

matches = [[0] for x in range(anime.shape[0])]
for category in categories:
    for genre, i in zip(genres, range(genres.shape[0])):
        if isinstance(genre, str):
            if category in genre:
                matches[i].append(1)
            else:
                matches[i].append(0)
        else:
                matches[i].append(0)

new_frame = anime["title"].str.lower()
new_matches = pd.DataFrame.from_records(matches)
new_matches = new_matches.drop(columns=0)
new_matches.columns = categories
data = pd.concat([new_frame, new_matches], axis=1)

data.to_csv('cleaned_data.csv', index=False)