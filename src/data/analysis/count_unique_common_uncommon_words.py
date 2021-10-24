import os.path
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm

from src.data.analysis.phase_one_analysis_abc import PhaseOneAnalysisABC


# noinspection DuplicatedCode
class CountCUCWords(PhaseOneAnalysisABC):
    @property
    def required_files(self) -> List[str]:
        return [
            os.path.join(self.data_dir, 'processed', 'tweets_word_broken.json')
        ]

    @property
    def report_name(self) -> str:
        return "count_cuc_words"

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
            ut_per_class[b] = set()
            uw_per_class[b] = set()
        tqdm.pandas()
        self.data.progress_apply(lambda r: ut_per_class[r['bias']].update(r['tokens']), axis=1)
        self.data.progress_apply(lambda r: uw_per_class[r['bias']].update(r['words']), axis=1)

        df_c_t = pd.DataFrame(columns=biases, dtype=float)
        df_c_t_p = pd.DataFrame(columns=biases, dtype=float)
        df_c_w = pd.DataFrame(columns=biases, dtype=float)
        df_c_w_p = pd.DataFrame(columns=biases, dtype=float)
        df_uc_t = pd.DataFrame(columns=biases, dtype=float)
        df_uc_t_p = pd.DataFrame(columns=biases, dtype=float)
        df_uc_w = pd.DataFrame(columns=biases, dtype=float)
        df_uc_w_p = pd.DataFrame(columns=biases, dtype=float)
        for b1 in biases:
            row_c_t = {b2: 0 for b2 in biases}
            row_c_t_p = {b2: 0 for b2 in biases}
            row_c_w = {b2: 0 for b2 in biases}
            row_c_w_p = {b2: 0 for b2 in biases}
            row_uc_t = {b2: 0 for b2 in biases}
            row_uc_t_p = {b2: 0 for b2 in biases}
            row_uc_w = {b2: 0 for b2 in biases}
            row_uc_w_p = {b2: 0 for b2 in biases}
            for b2 in biases:
                row_c_t[b2] = len(ut_per_class[b1].intersection(ut_per_class[b2]))
                row_c_w[b2] = len(uw_per_class[b1].intersection(uw_per_class[b2]))
                row_uc_t[b2] = len(ut_per_class[b1].symmetric_difference(ut_per_class[b2]))
                row_uc_w[b2] = len(uw_per_class[b1].symmetric_difference(uw_per_class[b2]))

                row_c_t_p[b2] = len(ut_per_class[b1].intersection(ut_per_class[b2])) / \
                    len(ut_per_class[b1].union(ut_per_class[b2])) * 100
                row_c_w_p[b2] = len(uw_per_class[b1].intersection(uw_per_class[b2])) / \
                    len(uw_per_class[b1].union(uw_per_class[b2])) * 100
                row_uc_t_p[b2] = len(ut_per_class[b1].symmetric_difference(ut_per_class[b2])) / \
                    len(ut_per_class[b1].union(ut_per_class[b2])) * 100
                row_uc_w_p[b2] = len(uw_per_class[b1].symmetric_difference(uw_per_class[b2])) / \
                    len(uw_per_class[b1].union(uw_per_class[b2])) * 100

            row_c_t = pd.Series(row_c_t, name=b1, dtype=float)
            row_c_t_p = pd.Series(row_c_t_p, name=b1, dtype=float)
            row_c_w = pd.Series(row_c_w, name=b1, dtype=float)
            row_c_w_p = pd.Series(row_c_w_p, name=b1, dtype=float)
            row_uc_t = pd.Series(row_uc_t, name=b1, dtype=float)
            row_uc_t_p = pd.Series(row_uc_t_p, name=b1, dtype=float)
            row_uc_w = pd.Series(row_uc_w, name=b1, dtype=float)
            row_uc_w_p = pd.Series(row_uc_w_p, name=b1, dtype=float)

            df_c_t = df_c_t.append(row_c_t)
            df_c_t_p = df_c_t_p.append(row_c_t_p)
            df_c_w = df_c_w.append(row_c_w)
            df_c_w_p = df_c_w_p.append(row_c_w_p)
            df_uc_t = df_uc_t.append(row_uc_t)
            df_uc_t_p = df_uc_t_p.append(row_uc_t_p)
            df_uc_w = df_uc_w.append(row_uc_w)
            df_uc_w_p = df_uc_w_p.append(row_uc_w_p)

        self._dfs = {
            'c_t': df_c_t.sort_index().sort_index(axis=1),
            'c_t_p': df_c_t_p.sort_index().sort_index(axis=1),
            'c_w': df_c_w.sort_index().sort_index(axis=1),
            'c_w_p': df_c_w_p.sort_index().sort_index(axis=1),
            'uc_t': df_uc_t.sort_index().sort_index(axis=1),
            'uc_t_p': df_uc_t_p.sort_index().sort_index(axis=1),
            'uc_w': df_uc_w.sort_index().sort_index(axis=1),
            'uc_w_p': df_uc_w_p.sort_index().sort_index(axis=1),
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
        for df_name, df in self._dfs.items():
            sns.set_theme('paper')
            c = (df.to_numpy().min() + df.to_numpy().max()) // 2
            ax = sns.heatmap(df, annot=True, fmt=".1f", center=c, cmap='coolwarm')
            plt.xlabel('Bias')
            plt.ylabel('Bias')
            ax.figure.savefig(os.path.join(self.report_dir, self.report_name, f'{df_name}.png'))
            plt.clf()
