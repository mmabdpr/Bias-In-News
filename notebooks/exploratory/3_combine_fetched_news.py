# %%
import pandas as pd

# %%
from os import listdir
from os.path import isfile, join
data_path = './data'
data_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# %%
all_articles_df = pd.DataFrame()
for f in data_files:
    df = pd.read_json(f'./data/{f}', typ='series', dtype=False)
    all_articles_df = all_articles_df.append(df, ignore_index=True)

# %%
all_articles_df.to_json('./fetched_news.json', orient='records', indent=2)
# all_articles_df[(all_articles_df['article'] == "") | (all_articles_df['headline'] == "")]
# all_articles_df[all_articles_df['article_id'] == 16814]
# all_articles_df