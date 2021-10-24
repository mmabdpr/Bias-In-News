import os.path
import re
from collections import Counter
from pathlib import Path
from typing import List
from tabulate import tabulate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data.analysis.phase_one_analysis_abc import PhaseOneAnalysisABC


# noinspection DuplicatedCode
class TopTenRNF(PhaseOneAnalysisABC):
    @property
    def required_files(self) -> List[str]:
        return [
            os.path.join(self.data_dir, 'processed', 'tweets_word_broken.json')
        ]

    @property
    def report_name(self) -> str:
        return "top_ten_rnf"

    # noinspection PyUnreachableCode
    def __init__(self, data_dir: str, report_dir: str, json_orient: str = 'records'):
        self.data_dir = data_dir
        self.report_dir = report_dir
        self.json_orient = json_orient

        self.check_for_required_files()

        self.data = pd.read_json(os.path.join(self.data_dir, 'processed', 'tweets_word_broken.json'),
                                 orient=self.json_orient)

        # self.data = self.data.sample(frac=0.01).reset_index(drop=True)

        Path(os.path.join(self.report_dir, self.report_dir, self.report_name)).mkdir(parents=True, exist_ok=True)

        self.n = 10
        self._prepare_df()

    def _prepare_df(self):
        biases = sorted(self.data['bias'].unique().tolist())
        ut_per_class = dict()
        uw_per_class = dict()
        for b in biases:
            ut_per_class[b] = Counter()
            uw_per_class[b] = Counter()

        tqdm.pandas()
        self.data.progress_apply(lambda r: ut_per_class[r['bias']].update(r['tokens']), axis=1)
        self.data.progress_apply(lambda r: uw_per_class[r['bias']].update(r['words']), axis=1)

        count_all_t = {b: sum(ut_per_class[b].values()) for b in biases}
        count_all_w = {b: sum(uw_per_class[b].values()) for b in biases}

        df_rnf_t = pd.DataFrame(columns=['Bias 1', 'Bias 2', 'RNFs'])
        df_rnf_w = pd.DataFrame(columns=['Bias 1', 'Bias 2', 'RNFs'])
        for b1 in biases:
            for b2 in biases:
                if b1 == b2:
                    continue

                row_rnf_t = {
                    'Bias 1': b1,
                    'Bias 2': b2,
                    'RNFs': {}
                }
                row_rnf_w = {
                    'Bias 1': b1,
                    'Bias 2': b2,
                    'RNFs': {}
                }

                common_tokens = set(ut_per_class[b1].keys()).intersection(set(ut_per_class[b2].keys()))
                common_words = set(uw_per_class[b1].keys()).intersection(set(uw_per_class[b2].keys()))

                for ct in common_tokens:
                    row_rnf_t['RNFs'][ct] = (ut_per_class[b1][ct] / count_all_t[b1]) / (ut_per_class[b2][ct] / count_all_t[b2])

                for cw in common_words:
                    row_rnf_w['RNFs'][cw] = (uw_per_class[b1][cw] / count_all_w[b1]) / (uw_per_class[b2][cw] / count_all_w[b2])

                row_rnf_t['RNFs'] = sorted(row_rnf_t['RNFs'].items(), key=lambda e: e[1], reverse=True)[:self.n]
                row_rnf_w['RNFs'] = sorted(row_rnf_w['RNFs'].items(), key=lambda e: e[1], reverse=True)[:self.n]

                row_rnf_t = pd.Series(row_rnf_t)
                row_rnf_w = pd.Series(row_rnf_w)

                df_rnf_t = df_rnf_t.append(row_rnf_t, ignore_index=True)
                df_rnf_w = df_rnf_w.append(row_rnf_w, ignore_index=True)

        self._dfs = {
            'rnf_t': df_rnf_t.sort_index().sort_index(axis=1),
            'rnf_w': df_rnf_w.sort_index().sort_index(axis=1),
        }

    def generate_tables(self):
        for i in range(self.n):
            for df_name, df in self._dfs.items():
                df['RNF'] = df['RNFs'].apply(lambda r: r[i])
                df[f'Rank {i}'] = df['Bias 1']
                df = df.pivot(index=f'Rank {i}', columns='Bias 2', values='RNF')
                df = df.replace(np.nan, '')
                latex_table = tabulate(df, headers='keys', tablefmt='latex')
                with open(os.path.join(self.report_dir, self.report_name, f'table_{df_name}_rank_{i}.latex.txt'), 'w') as f:
                    f.write(latex_table)
                grid_table = tabulate(df, headers='keys', tablefmt='grid')
                with open(os.path.join(self.report_dir, self.report_name, f'table_{df_name}_rank_{i}.grid.txt'), 'w') as f:
                    f.write(grid_table)

    def generate_graphs(self):
        p = re.compile(r'.* \(([0-9]+)\)')
        for i in range(self.n):
            for df_name, df in self._dfs.items():
                figure = plt.gcf()
                figure.set_size_inches(10, 6)
                df['RNF'] = df['RNFs'].apply(lambda r: f'{r[i][0]} ({int(r[i][1])})')
                df[f'Rank {i}'] = df['Bias 1']
                df = df.pivot(index=f'Rank {i}', columns='Bias 2', values='RNF')
                df = df.replace(np.nan, ' (0)')
                df_data = df.applymap(lambda s: int(p.search(s).group(1))).astype(int)
                sns.set_theme('paper')
                c = (df_data.to_numpy().min() + df_data.to_numpy().max()) // 2
                ax = sns.heatmap(df_data, annot=df, center=c, fmt='', cmap='coolwarm')
                plt.xlabel('Bias')
                plt.ylabel('Bias')
                plt.title(f'Relative Normalized Frequency (Rank {i})')
                ax.figure.savefig(os.path.join(self.report_dir, self.report_name, f'{df_name}_rank_{i}.png'), dpi=300)
                plt.clf()
