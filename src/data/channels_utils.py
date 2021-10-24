import re
from typing import *
import pandas as pd
# from dataenforce import Dataset # TODO data type check


class AllSidesUtils:
    def __init__(self, html_file_path: str):
        self.html_file_path = html_file_path
        with open(self.html_file_path, "r") as f:
            self.html_file = f.read()

    def extract_data_from_html(self) -> pd.DataFrame:
        channel_names_pattern = re.compile(r'<td class="views-field views-field-title source-title">\n.*<a href=.*>(.*\n?.*)</a>')
        channel_names: List[str] = channel_names_pattern.findall(self.html_file)
        channel_names = [name.replace('\n', '').replace(' ', '') for name in channel_names]

        bias_rating_pattern = re.compile(r'<td class="views-field views-field-field-bias-image">\n.*\n.*\n.*title="AllSides Media Bias Rating: (.*)"\n')
        bias_ratings: List[str] = bias_rating_pattern.findall(self.html_file)
        bias_ratings = [r.replace(' ', '') for r in bias_ratings]

        community_feedbacks_pattern = re.compile(r'<span class=agree>([0-9]+)</span>/<span class=disagree>([0-9]+)</span>')
        community_feedbacks = community_feedbacks_pattern.findall(self.html_file)
        community_feedbacks_agree, community_feedbacks_disagree = [], []
        for cf in community_feedbacks:
            community_feedbacks_agree.append(int(cf[0]))
            community_feedbacks_disagree.append(int(cf[1]))

        assert len(channel_names) == len(bias_ratings)
        assert len(channel_names) == len(community_feedbacks_agree)
        assert len(channel_names) == len(community_feedbacks_disagree)

        df = pd.DataFrame({
            'channel_names': channel_names,
            'bias_ratings': bias_ratings,
            'community_feedbacks_agree': community_feedbacks_agree,
            'community_feedbacks_disagree': community_feedbacks_disagree
        })

        return df


# noinspection SpellCheckingInspection
class AdFontesMediaUtils:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        with open(self.json_file_path, "r") as f:
            self.json_file = f.read()
        self._data = None
        self._channels_data = None

    def extract_data_from_json(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data.copy()

        df = pd.read_json(self.json_file_path, orient="records")

        if len(df['source_id'].unique() > len(df['source'].unique())):
            # 'Houston Chronicle' and 'Forward' channels cause this problem
            # we remove all rows from these two channels
            df["n"] = df["source"].apply(lambda x: df[df["source"] == x].iloc[0]["source_id"])
            problematic_channels = df[df["n"] != df["source_id"]]["source"].unique()
            df = df[~df["source"].isin(problematic_channels)]

        df = df.drop(columns=["n"])

        assert len(df['source_id'].unique() == len(df['source'].unique()))

        self._data = df
        return self._data.copy()

    def extract_channels_data_from_json(self) -> pd.DataFrame:
        if self._channels_data is not None:
            return self._channels_data.copy()

        news_df = self.extract_data_from_json()

        df = news_df.groupby('source_id').agg({
            'source': ['min'],
            'domain': ['min'],
            'reach': ['min'],
            'mediatype': ['min'],
            'article_count': ['min'],
            'bias': ['mean'],
            'quality': ['mean']
        })

        df.columns = ['source', 'domain', 'reach', 'mediatype', 'article_count', 'avg_bias', 'avg_quality']
        df['source_id'] = df.index

        self._channels_data = df
        return self._channels_data.copy()


class MediaBiasFactCheckUtils:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        with open(self.json_file_path, "r") as f:
            self.json_file = f.read()

    def extract_data_from_json(self) -> pd.DataFrame:
        df = pd.read_json(self.json_file_path, orient="records")
        df = df[df['bias'] != '']
        df['country'] = df['country'].apply(lambda c: "Unknown" if c == 'Unknown or Invalid Region' or c == '' else c)
        df['credibility'] = df['credibility'].apply(lambda c: "Unknown" if c == '' or c == 'N/A' else c)
        df['freporting'] = df['freporting'].apply(lambda f: "Unknown" if f == '' else f)
        df['traffic'] = df['traffic'].apply(lambda t: "Unknown" if t == '' or t == 'No Data' else t)
        return df
