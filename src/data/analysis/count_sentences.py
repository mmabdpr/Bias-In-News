import os.path
from pathlib import Path
from typing import List
from tabulate import tabulate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.data.analysis.phase_one_analysis_abc import PhaseOneAnalysisABC


class CountSentences(PhaseOneAnalysisABC):
    @property
    def required_files(self) -> List[str]:
        return [
            os.path.join(self.data_dir, 'processed', 'tweets_sent_broken.json')
        ]

    @property
    def report_name(self) -> str:
        return "count_sentences"

    def __init__(self, data_dir: str, report_dir: str, json_orient: str = 'records'):
        self.data_dir = data_dir
        self.report_dir = report_dir
        self.json_orient = json_orient

        self.check_for_required_files()

        self.data = pd.read_json(os.path.join(self.data_dir, 'processed', 'tweets_sent_broken.json'), orient=self.json_orient)

        Path(os.path.join(self.report_dir, self.report_dir, self.report_name)).mkdir(parents=True, exist_ok=True)

    def generate_tables(self):
        df = pd.DataFrame()
        df['sentence_count'] = self.data['sentences'].apply(len)
        df['bias'] = self.data['bias'].values
        df = df.groupby('bias').agg({
            'sentence_count': ['sum']
        })
        df['bias'] = df.index.values
        df.columns = ['Sentence Count', 'Bias']
        latex_table = tabulate(df[['Bias', 'Sentence Count']].to_dict(orient='records'), headers='keys', tablefmt='latex')
        with open(os.path.join(self.report_dir, self.report_name, 'table.latex.txt'), 'w') as f:
            f.write(latex_table)
        grid_table = tabulate(df[['Bias', 'Sentence Count']].to_dict(orient='records'), headers='keys', tablefmt='grid')
        with open(os.path.join(self.report_dir, self.report_name, 'table.grid.txt'), 'w') as f:
            f.write(grid_table)

    def generate_graphs(self):
        df = pd.DataFrame()
        df['sentence_count'] = self.data['sentences'].apply(len)
        df['bias'] = self.data['bias'].values
        df = df.groupby('bias').agg({
            'sentence_count': ['sum']
        })
        df['bias'] = df.index.values
        df.columns = ['sentence_count', 'bias']
        df['sentence_count'] = df['sentence_count'].apply(lambda c: round(c / 1000, 2))
        sns.set_theme('paper')
        ax = sns.barplot(data=df, x='bias', y='sentence_count', palette='coolwarm')
        ax.set(xlabel='Bias', ylabel='Sentence Count (K)')
        ax.figure.savefig(os.path.join(self.report_dir, self.report_name, 'count_sentences.png'))
        plt.clf()
