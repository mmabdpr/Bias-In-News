import copy
import os.path
import re
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm

from src.data.analysis.phase_one_analysis_abc import PhaseOneAnalysisABC


# noinspection DuplicatedCode
class TopTenUCWords(PhaseOneAnalysisABC):
    @property
    def required_files(self) -> List[str]:
        return [
            os.path.join(self.data_dir, 'processed', 'tweets_word_broken.json')
        ]

    @property
    def report_name(self) -> str:
        return "top_ten_uc_words"

    def __init__(self, data_dir: str, report_dir: str, json_orient: str = 'records'):
        self.data_dir = data_dir
        self.report_dir = report_dir
        self.json_orient = json_orient

        self.check_for_required_files()

        self.data = pd.read_json(os.path.join(self.data_dir, 'processed', 'tweets_word_broken.json'),
                                 orient=self.json_orient)

        Path(os.path.join(self.report_dir, self.report_dir, self.report_name)).mkdir(parents=True, exist_ok=True)

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

        uct_per_class = dict()
        ucw_per_class = dict()

        for b1 in biases:
            uct_per_class[b1] = copy.deepcopy(ut_per_class[b1])
            ucw_per_class[b1] = copy.deepcopy(uw_per_class[b1])
            for b2 in biases:
                if b1 == b2:
                    continue

                common_keys = set(uct_per_class[b1].keys()).intersection(set(ut_per_class[b2].keys()))
                for k in common_keys:
                    del uct_per_class[b1][k]

                common_keys = set(ucw_per_class[b1].keys()).intersection(set(uw_per_class[b2].keys()))
                for k in common_keys:
                    del ucw_per_class[b1][k]

        n = 10
        most_common_tokens = dict()
        most_common_words = dict()
        for b in biases:
            most_common_tokens[b] = uct_per_class[b].most_common(n)
            most_common_words[b] = ucw_per_class[b].most_common(n)

        df_t = pd.DataFrame(columns=sorted(biases))
        df_w = pd.DataFrame(columns=sorted(biases))
        for i in range(n):
            row_t = dict()
            row_w = dict()
            for b in biases:
                row_t[b] = f'{most_common_tokens[b][i][0]} ({most_common_tokens[b][i][1]})'
                row_w[b] = f'{most_common_words[b][i][0]} ({most_common_words[b][i][1]})'
            row_t = pd.Series(row_t, name=i)
            row_w = pd.Series(row_w, name=i)
            df_t = df_t.append(row_t)
            df_w = df_w.append(row_w)

        self._dfs = {
            'token': df_t,
            'word': df_w
        }

    def generate_tables(self):
        for df_name, df in self._dfs.items():
            latex_table = tabulate(df, headers='keys', tablefmt='latex')
            with open(os.path.join(self.report_dir, self.report_name, f'table_{df_name}.latex.txt'), 'w') as f:
                f.write(latex_table)
            grid_table = tabulate(df, headers='keys', tablefmt='grid')
            with open(os.path.join(self.report_dir, self.report_name, f'table_{df_name}.grid.txt'), 'w') as f:
                f.write(grid_table)

    def generate_graphs(self):
        p = re.compile(r'.* \(([0-9]+)\)')
        for df_name, df in self._dfs.items():
            figure = plt.gcf()
            figure.set_size_inches(16, 9)
            df_data = df.applymap(lambda s: int(p.search(s).group(1))).astype(int)
            sns.set_theme('paper')
            c = (df_data.to_numpy().min() + df_data.to_numpy().max()) // 2
            ax = sns.heatmap(df_data, annot=df, center=c, fmt='', cmap='coolwarm')
            plt.xlabel('Bias')
            plt.ylabel('Rank')
            ax.figure.savefig(os.path.join(self.report_dir, self.report_name, f'{df_name}.png'), dpi=300)
            plt.clf()
