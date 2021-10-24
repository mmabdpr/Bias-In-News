# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
data_dir = '../../data'

# %%
df = pd.read_json(f'{data_dir}/fetched_news.json')

# %%
df = df[(df['article'] != "") & (df['headline'] != "")]

# %%
df_original = pd.read_json(f'{data_dir}/adfontesmedia.json')

# %%
df = df.merge(df_original, on='article_id')

# %%
df['source'].unique().shape, df['source_id'].unique().shape

# %%
_s = df['source'].unique()
df[df['source'].isin(_s)]

# %%
tmp = df.groupby('source_id').agg({'source_id': ['count']}).reset_index()
tmp.columns = ['source_id', 'count']
tmp = tmp.sort_values('count', ascending=False)
tmp = tmp[:80]
tmp

# %%
df[df['source_id'].isin(tmp['source_id'])]

# %%
boundaries = [-42, -9, -1.79, 1.79, 12, 42]

def get_partisanship(bias):
    for i, _ in enumerate(boundaries[:-1]):
        b1, b2 = boundaries[i], boundaries[i+1]
        if b1 <= bias < b2:
            return i - len(boundaries) // 2

tmp = df.groupby('source_id').agg({'bias': ['mean']}).reset_index()
tmp.columns = ['source_id', 'bias_mean']
tmp.set_index('source_id', inplace=True)

arr = plt.hist(tmp['bias_mean'], alpha=0.5, bins=boundaries)
for i in range(5):
    plt.text(arr[1][i] + 3, arr[0][i] + 5, str(int(arr[0][i])))
plt.show()

tmp['channel_partisanship'] = tmp['bias_mean'].apply(get_partisanship)

arr = plt.hist(tmp['channel_partisanship'], alpha=0.5, bins=5)
for i in range(5):
    plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))
plt.show()


# %%
tmp = df.groupby('source').agg({'bias': ['mean', 'std'], 'article_count': ['mean'], 'quality': ['mean'], 'reach': ['mean'], 'url': ['min']}).reset_index()
tmp.columns = ['source', 'bias_mean', 'bias_std', 'article_count', 'quality', 'reach_M', 'url']
tmp['reach_M'] /= 1_000_000
tmp = tmp.sort_values(by=['bias_mean'])


# %%
df['article_partisanship'] = df['bias'].apply(get_partisanship)
df

# %%
# select top 10 channels from each zone
# sort by article count
# filter out low quality

df['channel_partisanship'] = df['source_id'].apply(lambda id: tmp.loc[id]['channel_partisanship'])
df

# %%
df_bad = df[abs(df['article_partisanship'] - df['channel_partisanship']) >= 1]
df_bad = df_bad.groupby('source').agg({'source': ['count']}).reset_index()
df_bad.columns = ['source', 'count']
df_bad = df_bad.sort_values('count', ascending=False)
df_bad

# %%
_g = df.groupby('source').agg({'article_partisanship': 'std'}).reset_index()
_g.columns = ['source', 'ap_std']
_g.sort_values(by=['ap_std'], ascending=False)[:40]

# %%
df['channel_partisanship'].corr(df['article_partisanship'])
# 0.793

# %%
df_fl = df[df['channel_partisanship'] == -2]
df_cl = df[df['channel_partisanship'] == -1]
df_nt = df[df['channel_partisanship'] == 0]
df_cr = df[df['channel_partisanship'] == 1]
df_fr = df[df['channel_partisanship'] == 2]

# %%
df_fl['source'].unique()

# %%
g = df_cl.groupby('source_id').agg({'reach': ['mean'], 'bias': ['std', 'mean']}).reset_index()
g.columns = ['source_id', 'reach', 'bias_std', 'bias_mean']
cl_src = g.sort_values(by=['reach'], ascending=False)[:10]['source_id']

df_cl_slc_by_rc = df_cl[df_cl['source_id'].isin(cl_src)]
df_cl_slc_by_rc


# %%
g = df_cl.groupby('source_id').agg({'article_count': ['mean'], 'bias': ['std', 'mean']}).reset_index()
g.columns = ['source_id', 'article_count', 'bias_std', 'bias_mean']
cl_src = g.sort_values(by=['article_count'], ascending=False)[:10]['source_id']

df_cl_slc_by_ac = df_cl[df_cl['source_id'].isin(cl_src)]
df_cl_slc_by_ac

# %%
ids_1 = df_cl_slc_by_rc['source'].unique()
ids_1
# %%
ids_2 = df_cl_slc_by_ac['source'].unique()
ids_2

# %%
ids = set().union(ids_1, ids_2)
ids, len(ids)

# %%
df_cl_slc_merged = df_cl[df_cl['source'].isin(ids)]
_g = df_cl_slc_merged.groupby('source').agg({'quality': ['mean'], 'bias': ['std', 'mean']})
_g.columns = ['quality', 'bias_std', 'bias_mean']
_g.sort_values(by=['bias_mean'])

# %%
g = df_nt.groupby('source_id').agg({'reach': ['mean'], 'bias': ['mean']}).reset_index()
g.columns = ['source_id', 'reach', 'bias']

nt_src_l = g[g['bias'] >= 0].sort_values(by=['reach'], ascending=False)[:20]['source_id']
nt_src_r = g[g['bias'] < 0].sort_values(by=['reach'], ascending=False)[:20]['source_id']

print(nt_src_l.unique().shape, nt_src_r.unique().shape)

nt_src = nt_src_r.append(nt_src_l)

df_nt_slc_by_rc = df_nt[df_nt['source_id'].isin(nt_src)]
df_nt_slc_by_rc


# %%
g = df_nt.groupby('source_id').agg({'article_count': ['mean'], 'bias': ['mean']}).reset_index()
g.columns = ['source_id', 'article_count', 'bias']

nt_src_l = g[g['bias'] >= 0].sort_values(by=['article_count'], ascending=False)[:20]['source_id']
nt_src_r = g[g['bias'] < 0].sort_values(by=['article_count'], ascending=False)[:20]['source_id']

print(nt_src_l.unique().shape, nt_src_r.unique().shape)

df_nt_slc_by_ac = df_nt[df_nt['source_id'].isin(nt_src)]
df_nt_slc_by_ac

# %%
ids_1 = df_nt_slc_by_rc['source'].unique()
ids_1
# %%
ids_2 = df_nt_slc_by_ac['source'].unique()
ids_2

# %%
ids = set().union(ids_1, ids_2)
ids, len(ids)

# %%
df_nt_slc_merged = df_nt[df_nt['source'].isin(ids)]
_g = df_nt_slc_merged.groupby('source').agg({'quality': ['mean'], 'bias': ['std', 'mean']})
_g.columns = ['quality', 'bias_std', 'bias_mean']
_g.sort_values(by=['bias_mean'])

# %%
['CNN', 'NBC News', 'The New York Times', 'Washington Post', 'ABC News', 
'CNET', 'Reuters', 'AP', 'The Hill', 'The Economist', 'Sky', 'Napa Valley Register']

# %%
df[df['source'].str.contains('Sky')]


# %%
g = df_fl.groupby('source_id').agg({'reach': ['mean'], 'bias': ['mean']}).reset_index()
g.columns = ['source_id', 'reach', 'bias']

fl_src_l = g[g['bias'] >= 0].sort_values(by=['reach'], ascending=False)[:20]['source_id']
fl_src_r = g[g['bias'] < 0].sort_values(by=['reach'], ascending=False)[:20]['source_id']

print(fl_src_l.unique().shape, fl_src_r.unique().shape)

fl_src = fl_src_r.append(fl_src_l)

df_fl_slc_by_rc = df_fl[df_fl['source_id'].isin(fl_src)]

ids = df_fl_slc_by_rc['source'].unique()

df_fl_slc_merged = df_fl[df_fl['source'].isin(ids)]
_g = df_fl_slc_merged.groupby('source').agg({'quality': ['mean'], 'bias': ['std', 'mean']})
_g.columns = ['quality', 'bias_std', 'bias_mean']
_g.sort_values(by=['bias_mean'])
