# %%
from newsfetch.news import newspaper
import pandas as pd
import os
import multiprocessing
import time
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired


# %%
df = pd.read_json(r'sourceData.json')
df['article_id'].shape

# %%
def fetch_news(url):
    return newspaper(url)

for index, row in df.iterrows():
    try:
        if os.path.isfile(f"./data/{row['article_id']}.json"):
            print(f'already have {row["article_id"]}')
            continue
        
        with ProcessPool() as pool:
            future = pool.map(fetch_news, [row['url']], timeout=20)

            iterator = future.result()

            while True:
                try:
                    result = next(iterator)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(f"r{row['article_id']} function took longer than {error.args[1]} seconds")
                except ProcessExpired as error:
                    print(f"r{row['article_id']} {error}. Exit code: {error.exitcode}")
                except Exception as error:
                    print(f"r{row['article_id']} function raised {error}")
                    print(error.traceback)

        article = result.article
        headline = result.headline
        news_df = pd.Series({'article_id': row['article_id'], 'article': article, 'headline': headline})
        news_df.to_json(f"./data/{row['article_id']}.json")
    except Exception as e:
        with open('./errors.txt', 'a') as f:
            f.write(f"{row['article_id']}\t{2}\n")

# %%
