import logging
import os

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

from src.data.analysis.count_sentences import CountSentences
from src.data.analysis.count_tweets import CountTweets
from src.data.analysis.count_unique_common_uncommon_words import CountCUCWords
from src.data.analysis.count_unique_words import CountUniqueWords
from src.data.analysis.count_words import CountWords
from src.data.analysis.top_ten_rnf import TopTenRNF
from src.data.analysis.top_ten_tf_idf import TopTenTFIDF
from src.data.analysis.top_ten_uc_words import TopTenUCWords
from src.data.analysis.words_histogram import WordsHistogram

project_dir = Path(__file__).resolve().parents[2]
data_dir = os.path.join(project_dir, "data")
report_dir = os.path.join(project_dir, 'docs', 'phase_1_report', 'figs')


def count_tweets():
    ct = CountTweets(data_dir=data_dir, report_dir=report_dir)
    ct.generate_tables()
    ct.generate_graphs()


def count_sentences():
    cs = CountSentences(data_dir=data_dir, report_dir=report_dir)
    cs.generate_tables()
    cs.generate_graphs()


def count_words():
    cw = CountWords(data_dir=data_dir, report_dir=report_dir)
    cw.generate_tables()
    cw.generate_graphs()


def count_unique_words():
    cw = CountUniqueWords(data_dir=data_dir, report_dir=report_dir)
    cw.generate_tables()
    cw.generate_graphs()


def count_unique_common_uncommon_words():
    cw = CountCUCWords(data_dir=data_dir, report_dir=report_dir)
    cw.generate_tables()
    cw.generate_graphs()


def get_top_ten_uc_words():
    cw = TopTenUCWords(data_dir=data_dir, report_dir=report_dir)
    cw.generate_tables()
    cw.generate_graphs()


def get_top_ten_rnf():
    cw = TopTenRNF(data_dir=data_dir, report_dir=report_dir)
    cw.generate_tables()
    cw.generate_graphs()


def get_top_ten_tf_idf():
    cw = TopTenTFIDF(data_dir=data_dir, report_dir=report_dir)
    cw.generate_tables()
    cw.generate_graphs()


def get_histogram_of_words():
    cw = WordsHistogram(data_dir=data_dir, report_dir=report_dir)
    cw.generate_tables()
    cw.generate_graphs()


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    # TODO add total, average, std, .. rows
    # TODO remove hardcoded stop words and use file
    # TODO iteratively add to stop words
    # TODO add log statements
    count_tweets()
    count_sentences()
    count_words()
    count_unique_words()
    count_unique_common_uncommon_words()
    get_top_ten_uc_words()
    get_top_ten_rnf()
    get_top_ten_tf_idf()
    get_histogram_of_words()
