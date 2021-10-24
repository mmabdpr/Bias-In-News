import os.path
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm

from src.data.analysis.phase_one_analysis_abc import PhaseOneAnalysisABC


# noinspection DuplicatedCode
class CountUniqueWords(PhaseOneAnalysisABC):
    @property
    def required_files(self) -> List[str]:
        return [
            os.path.join(self.data_dir, 'processed', 'tweets_word_broken.json')
        ]

    @property
    def report_name(self) -> str:
        return "count_unique_words"

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
        df = pd.DataFrame(columns=['Bias', 'Token Count', 'Word Count'], dtype=int)
        for b in biases:
            df = df.append({'Bias': b, 'Token Count': len(ut_per_class[b]), 'Word Count': len(uw_per_class[b])}, ignore_index=True)
        self._df = df.sort_values(by='Bias', ascending=True)

    def generate_tables(self):
        df = self._df
        latex_table = tabulate(df.to_dict(orient='records'), headers='keys', tablefmt='latex')
        with open(os.path.join(self.report_dir, self.report_name, 'table.latex.txt'), 'w') as f:
            f.write(latex_table)
        grid_table = tabulate(df.to_dict(orient='records'), headers='keys', tablefmt='grid')
        with open(os.path.join(self.report_dir, self.report_name, 'table.grid.txt'), 'w') as f:
            f.write(grid_table)

    def generate_graphs(self):
        df = pd.melt(self._df, id_vars=['Bias'], value_vars=['Token Count', 'Word Count'], var_name='Variable',
                     value_name='Count')

        sns.set_theme('paper')
        palette = plt.get_cmap("coolwarm")

        def rescale(y):
            return (y - np.min(y)) / (np.max(y) - np.min(y))

        p1 = plt.bar(data=df[df['Variable'] == 'Token Count'], x='Bias', height='Count', width=-0.4, align='edge',
                     color=palette(rescale(df['Bias'])), hatch='')
        p2 = plt.bar(data=df[df['Variable'] == 'Word Count'], x='Bias', height='Count', width=0.4, align='edge',
                     color=palette(rescale(df['Bias'])), hatch='/')
        plt.legend([p1, p2], ['Token', 'Word'], prop={'size': 12})
        plt.xlabel('Bias')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.report_dir, self.report_name, 'count_tokens_words.png'))
        plt.clf()
