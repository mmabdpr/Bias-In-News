# %%
import pandas as pd
import os
import difflib
from src.data.channels_utils import AllSidesUtils, AdFontesMediaUtils, MediaBiasFactCheckUtils
from typing import *
from pathlib import Path

# %%
data_dir = 'data'

util_allsides = AllSidesUtils(html_file_path=os.path.join(data_dir, "intermediate", "allsides.html"))
allsides_data = util_allsides.extract_data_from_html()

util_adfontes = AdFontesMediaUtils(json_file_path=os.path.join(data_dir, "intermediate", "adfontesmedia.json"))
adfontesmedia_data = util_adfontes.extract_channels_data_from_json()

util_mediabias = MediaBiasFactCheckUtils(
    json_file_path=os.path.join(data_dir, "intermediate", "mediabiasfactcheck.json"))
mediabiasfactcheck_data = util_mediabias.extract_data_from_json()

# %%
allsides_channel_names: List[str] = allsides_data['channel_names'].unique().tolist()
adfontes_channel_names: List[str] = adfontesmedia_data['source'].unique().tolist()
mediabias_channel_names: List[str] = mediabiasfactcheck_data['name'].unique().tolist()


def get_match_safe(row, candidates, name='', col_name='channel_names'):
    word = row[col_name]
    matches = difflib.get_close_matches(word, candidates, n=3, cutoff=0.3)
    match_dict = {
        f'{name}_match_0': '',
        f'{name}_match_1': '',
        f'{name}_match_2': '',
    }

    for i, m in enumerate(matches):
        match_dict[f'{name}_match_{i}'] = m

    for k, v in match_dict.items():
        row[k] = v

    return row


# ##
allsides_match = pd.DataFrame()
allsides_match["allsides_name"] = allsides_data["channel_names"]
allsides_match = allsides_match.apply(lambda r: get_match_safe(r, adfontes_channel_names, 'adfontes', 'allsides_name'), axis=1)
allsides_match = allsides_match.apply(lambda r: get_match_safe(r, mediabias_channel_names, 'mediabias', 'allsides_name'), axis=1)

# %%
adfontes_match = pd.DataFrame()
adfontes_match["adfontes_name"] = adfontesmedia_data["source"]
adfontes_match = adfontes_match.apply(lambda r: get_match_safe(r, allsides_channel_names, 'allsides', 'adfontes_name'), axis=1)
adfontes_match = adfontes_match.apply(lambda r: get_match_safe(r, mediabias_channel_names, 'mediabias', 'adfontes_name'), axis=1)

