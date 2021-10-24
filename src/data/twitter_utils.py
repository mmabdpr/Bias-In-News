import logging
import os
from pathlib import Path
from typing import *

import pandas as pd
import tweepy


def get_twitter_handles(channels_file_path: str) -> List[str]:
    df = pd.read_json(channels_file_path, typ='frame', orient='records', dtype=False)
    return df[df['twitter'] != "?"]['twitter'].tolist()


def get_twitter_handles_from_matched_channels(file_path: str) -> List[str]:
    df = pd.read_csv(file_path, sep='\t')
    handles: List[str] = df['twitter_id'].unique().tolist()
    handles.remove('<>')
    return handles


def get_headlines_from_twitter_timeline(twitter_handles: List[str],
                                        api_tokens: Dict[str, str],
                                        data_path: str):
    logger = logging.getLogger(__name__)
    auth = tweepy.OAuthHandler(api_tokens["consumer_key"], api_tokens["consumer_secret"])
    auth.set_access_token(api_tokens["access_token"], api_tokens["access_token_secret"])
    api = tweepy.API(auth, wait_on_rate_limit=True)
    Path(data_path).mkdir(parents=True, exist_ok=True)
    for handle in twitter_handles:
        try:
            file_name = os.path.join(data_path, f'{handle}.json')
            if os.path.isfile(file_name):
                logger.info(f'file for handle {handle} exists. skipping...')
                continue
            logger.info(f'getting tweets of {handle}')
            tweets = []
            for tweet in tweepy.Cursor(api.user_timeline, id=handle, tweet_mode='extended', include_rts=False, lang='en').items():
                tweets.append(tweet.full_text)
            df = pd.DataFrame(tweets, columns=['tweet'])
            df.to_json(file_name, indent=2, orient='records')
            logger.info(f"saved tweets of {handle}")
        except Exception as e:
            logger.error(f"error getting data for handle {handle} \n--\n {e}")
