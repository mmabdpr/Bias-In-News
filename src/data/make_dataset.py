import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from src.data.dataset_utils import combine_all_tweets, clean_tweets, \
    word_break_tweets, sentence_break_tweets, remove_popularized_words
from src.data.twitter_utils import get_headlines_from_twitter_timeline, get_twitter_handles_from_matched_channels

project_dir = Path(__file__).resolve().parents[2]
data_dir = os.path.join(project_dir, "data")

required_files = {
    "allsides.html": os.path.join(data_dir, 'intermediate', 'allsides.html'),
    "adfontesmedia.json": os.path.join(data_dir, 'intermediate', 'adfontesmedia.json'),
    "mediabiasfactcheck.json": os.path.join(data_dir, 'intermediate', 'mediabiasfactcheck.json'),
}


def check_for_file(name: str, path: str):
    if not os.path.isfile(path):
        logging.error(f"required file {name} was not found in path {path}")
        raise FileNotFoundError(f"required file {name} was not found in path {path}")


def check_for_required_files(msg=''):
    def check_process():
        logging.info("checking for required files")
        for rf_name, rf_path in required_files.items():
            check_for_file(rf_name, rf_path)
        logging.info('checks for required files passed ✅')

    if not msg.isspace():
        while True:
            input(msg)
            try:
                check_process()
                break
            except FileNotFoundError:
                print('try again')
    else:
        check_process()


def get_twitter_headlines():
    logging.info("getting twitter headlines")

    twitter_handles = get_twitter_handles_from_matched_channels(
        os.path.join(data_dir, "intermediate", "matched_channels.tsv"))

    api_tokens = {
        "consumer_key": os.getenv("TWITTER_API_CONSUMER_KEY"),
        "consumer_secret": os.getenv("TWITTER_API_CONSUMER_SECRET"),
        "access_token": os.getenv("TWITTER_API_ACCESS_TOKEN"),
        "access_token_secret": os.getenv("TWITTER_API_ACCESS_TOKEN_SECRET"),
    }

    get_headlines_from_twitter_timeline(twitter_handles, api_tokens,
                                        data_path=os.path.join(data_dir, "raw", "twitter_headlines_timeline"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    check_for_required_files(msg="make sure required files exist then press enter")

    required_files['matched_channels.tsv'] = os.path.join(data_dir, 'intermediate', 'matched_channels.tsv')
    check_for_required_files(msg="run select_top_channels notebook. after analyzing matched channels create file 'data/intermediate/matched_channels.tsv' with 4 columns 'allsides', 'adfontes', 'mediabias', 'twitter_id'. then press enter.")

    get_twitter_headlines()

    required_files['biases.tsv'] = os.path.join(data_dir, 'intermediate', 'biases.tsv')
    check_for_required_files(msg="run decide_biases notebook. after analyzing biases from all sources create a file 'data/intermediate/biases.tsv with 4 columns 'allsides', 'adfontes', 'mediabias', 'twitter_id'")

    combine_all_tweets()  # artifact: 'data/intermediate/tweets.json' with 3 columns 'tweet', 'source', 'bias'
    clean_tweets()  # artifact: 'data/processed/tweets.json' with 3 columns 'tweet', 'source', 'bias'
    sentence_break_tweets()  # artifact: 'data/processed/tweets_sent_broken.json' with 4 columns 'tweet', 'sentences', 'source', 'bias'
    word_break_tweets()  # artifact: 'data/processed/tweets_word_broken.json' with 5 columns 'tweet', 'tokens', 'words', 'source', 'bias'
    remove_popularized_words()  # updates 'data/processed/tweets_word_broken.json'
