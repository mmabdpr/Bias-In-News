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
matched_df = pd.read_csv(os.path.join(data_dir, 'intermediate', 'matched_channels.tsv'), sep='\t')


def add_bias_cols(row):
    name = row['allsides']
    if pd.notna(name):
        row['allsides_bias'] = allsides_data[allsides_data['channel_names'] == name].iloc[0]['bias_ratings']
    else:
        row['allsides_bias'] = 'Unknown'

    name = row['adfontes']
    if pd.notna(name):
        row['adfontes_bias'] = adfontesmedia_data[adfontesmedia_data['source'] == name].iloc[0]['avg_bias']
    else:
        row['adfontes_bias'] = 'Unknown'

    name = row['mediabias']
    if pd.notna(name):
            row['mediabias_bias'] = mediabiasfactcheck_data[mediabiasfactcheck_data['name'] == name].iloc[0]['bias']
    else:
        row['mediabias_bias'] = 'Unknown'

    return row


matched_df = matched_df.apply(add_bias_cols, axis=1)
