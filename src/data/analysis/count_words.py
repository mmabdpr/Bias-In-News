import os.path
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from src.data.analysis.phase_one_analysis_abc import PhaseOneAnalysisABC


# noinspection DuplicatedCode
class CountWords(PhaseOneAnalysisABC):
    @property
    def required_files(self) -> List[str]:
        return [
            os.path.join(self.data_dir, 'processed', 'tweets_word_broken.json')
        ]

    @property
    def report_name(self) -> str:
        return "count_words"

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
        df = pd.DataFrame()
        df['token_count'] = self.data['tokens'].apply(len)
        df['word_count'] = self.data['words'].apply(len)
        df['bias'] = self.data['bias'].values
        df = df.groupby('bias').agg({
            'token_count': ['sum'],
            'word_count': ['sum'],
        })
        df['bias'] = df.index.values
        df.reset_index(drop=True, inplace=True)
        df.columns = ['Token Count', 'Word Count', 'Bias']
        df = df[['Bias', 'Token Count', 'Word Count']]
        self._df = df

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
        plt.ylabel('Count (M)')
        plt.savefig(os.path.join(self.report_dir, self.report_name, 'count_tokens_words.png'))
        plt.clf()
