import nltk
import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
from typing import *
from tqdm import tqdm
# noinspection PyPackageRequirements
import preprocessor as p
from nltk.corpus import stopwords

project_dir = Path(__file__).resolve().parents[2]
data_dir = os.path.join(project_dir, "data")

json_orient = "records"


def get_class_labels():  # TODO infer from data
    return [-2, -1, 0, 1, 2]


# TODO add force redo option for all relevant functions

def get_tweets_with_source() -> pd.DataFrame:
    """twitter handles are based on data/intermediate/matched_channels.tsv file"""
    matched_df = pd.read_csv(os.path.join(data_dir, 'intermediate', 'matched_channels.tsv'), sep='\t')
    tweets = pd.DataFrame(columns=['tweet', 'source'])
    sources = matched_df[matched_df['twitter_id'] != '<>']['twitter_id'].unique().tolist()

    for source in sources:
        df = pd.read_json(os.path.join(data_dir, 'raw', 'twitter_headlines_timeline', f"{source}.json"), orient='records')
        df['source'] = df['tweet'].apply(lambda x: source)
        tweets = tweets.append(df, ignore_index=True)

    return tweets


def get_sources_with_bias() -> Dict[str, int]:
    biases_df = pd.read_csv(os.path.join(data_dir, 'intermediate', 'biases.tsv'), sep='\t')
    biases_dict = dict(pd.Series(biases_df['overall_bias'].values, index=biases_df['twitter_id'].values).to_dict(into=dict))
    biases_dict.pop('<>')
    return biases_dict


def combine_all_tweets():
    path = os.path.join(data_dir, 'intermediate', 'tweets.json')
    if os.path.isfile(path):
        logging.info("file intermediate/tweets.json exists. skipping..")
        return
    logging.info("combining all tweets into one file")
    tweets = get_tweets_with_source()  # DataFrame['tweet', 'source']
    biases = get_sources_with_bias()  # Dict['source': 'bias']
    tqdm.pandas()
    tweets['bias'] = tweets['source'].progress_apply(lambda x: biases[x])
    tweets.to_json(path, indent=2, orient=json_orient)
    logging.info("saved all tweets into file data/intermediate/tweets.json")


# TODO replace "&amp;" and "&gt;"
# TODO remove empty tweets
# TODO remove [thread] [to] [ via getty] [video] (all cases)
def clean_tweets():
    path = os.path.join(data_dir, 'processed', 'tweets.json')
    if os.path.isfile(path):
        logging.info("file processed/tweets.json exists. skipping..")
        return
    logging.info("cleaning tweets")
    tweets = pd.read_json(os.path.join(data_dir, 'intermediate', 'tweets.json'), orient=json_orient)
    tqdm.pandas()
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)
    tweets['tweet'] = tweets['tweet'].progress_apply(p.clean)
    tweets.to_json(path, indent=2, orient=json_orient)
    logging.info("saved cleaned tweets into file data/processed/tweets.json")


def word_break_tweets():
    path = os.path.join(data_dir, 'processed', 'tweets_word_broken.json')
    if os.path.isfile(path):
        logging.info("file tweets_word_broken.json exists. skipping..")
        return
    logging.info("breaking tweets into words")
    tweets = pd.read_json(os.path.join(data_dir, 'processed', 'tweets.json'), orient=json_orient)
    stop_words = set(stopwords.words('english'))
    tqdm.pandas()
    # tt = TweetTokenizer()
    logging.info("tokenizing")
    tweets['tokens'] = tweets['tweet'].progress_apply(nltk.word_tokenize)
    logging.info("removing stop words")
    tweets['tokens'] = tweets['tokens'].progress_apply(lambda tokens: [t.lower() for t in tokens if t.lower() not in stop_words])
    logging.info("extracting words (non-alpha)")
    tweets['words'] = tweets['tokens'].progress_apply(lambda tokens: [t for t in tokens if t.isalpha()])
    tweets.to_json(path, indent=2, orient=json_orient)
    logging.info("saved word broken tweets into file data/processed/tweets_word_broken.json")


def remove_popularized_words():
    path = os.path.join(data_dir, 'processed', 'tweets_word_broken.json')
    logging.info("removing artificially popularized words")
    tweets = pd.read_json(path, orient=json_orient)

    for tw in ['word', 'token']:
        tws: Dict[str: Set] = {}

        for i, row in tqdm(tweets.iterrows(), total=tweets.shape[0], desc=f'finding sources of each {tw}'):
            wl: List[str] = row[f'{tw}s']
            src = row['source']
            for w in wl:
                if w not in tws.keys():
                    tws[w] = set()
                tws[w].add(src)

        tws_to_be_removed = {w for w, s in tws.items() if len(s) <= 3}

        tqdm.pandas()
        tweets[f'{tw}s'] = tweets[f'{tw}s'].progress_apply(lambda l: [t for t in l if t not in tws_to_be_removed])

    tweets.to_json(path, indent=2, orient=json_orient)
    logging.info("saved updated word broken tweets into file data/processed/tweets_word_broken.json")


def sentence_break_tweets():
    path = os.path.join(data_dir, 'processed', 'tweets_sent_broken.json')
    if os.path.isfile(path):
        logging.info("file processed/tweets_sent_broken.json exists. skipping..")
        return
    logging.info("breaking tweets into sentences")
    tweets = pd.read_json(os.path.join(data_dir, 'processed', 'tweets.json'), orient=json_orient)
    tqdm.pandas()
    tweets['sentences'] = tweets['tweet'].progress_apply(nltk.sent_tokenize)
    tweets.to_json(path, indent=2, orient=json_orient)
    logging.info("saved sentence broken tweets into file data/processed/tweets_sent_broken.json")


def generate_small_dataset(n_per_class=1000):
    path = os.path.join(data_dir, 'processed', 'tweets_small.json')
    if os.path.isfile(path):
        logging.info("file processed/tweets_small.json exists. skipping..")
        return
    logging.info("generating small dataset")
    tweets = pd.read_json(os.path.join(data_dir, 'processed', 'tweets.json'), orient=json_orient)

    small_dataset = pd.DataFrame(columns=tweets.columns)
    for cls in tweets['bias'].unique():
        tweets_per_class = tweets[tweets['bias'] == cls].sample(n=n_per_class, random_state=42)
        small_dataset = small_dataset.append(tweets_per_class)

    small_dataset.to_json(path, indent=2, orient=json_orient)
    logging.info("saved small dataset into file data/processed/tweets_small.json")  # TODO use interpolation for file paths in log statements


def split_dataset(data: pd.DataFrame, train_frac=0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_train = pd.DataFrame(columns=data.columns)
    all_dev = pd.DataFrame(columns=data.columns)
    all_test = pd.DataFrame(columns=data.columns)

    for cls in data['bias'].unique():
        data_ = data[data['bias'] == cls].sample(frac=1., random_state=42)
        # noinspection PyTypeChecker
        train, dev, test = np.split(data_, [int(train_frac * len(data_)), int((train_frac + (1 - train_frac) / 2) * len(data_))])
        all_train = all_train.append(train)
        all_dev = all_dev.append(dev)
        all_test = all_test.append(test)

    return all_train, all_dev, all_test


def create_corpus(n_per_class=-1):
    corpus_dir = os.path.join(data_dir, 'processed', 'corpus')
    if not os.path.isdir(corpus_dir):
        os.mkdir(corpus_dir)

    tweets = pd.read_json(os.path.join(data_dir, 'processed', 'tweets.json'), orient=json_orient)
    classes = tweets['bias'].unique().tolist()
    for cls in classes:
        path = os.path.join(corpus_dir, f'tweets_corpus_{cls}.txt')
        if os.path.isfile(path):
            logging.info(f"file {path} exists. skipping..")
            continue
        if n_per_class == -1:
            t = tweets[tweets['bias'] == cls].sample(frac=1., random_state=42)
        else:
            t = tweets[tweets['bias'] == cls].sample(n=n_per_class, random_state=42)
        t = t[['tweet']]
        t.to_csv(path, header=False, index=False, line_terminator='\n\n')
