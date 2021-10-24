import os.path
from collections import Counter
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm

from src.data.analysis.phase_one_analysis_abc import PhaseOneAnalysisABC


# noinspection DuplicatedCode
class TopTenTFIDF(PhaseOneAnalysisABC):
    @property
    def required_files(self) -> List[str]:
        return [
            os.path.join(self.data_dir, 'processed', 'tweets_word_broken.json')
        ]

    @property
    def report_name(self) -> str:
        return "top_ten_tf_idf"

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

        df = pd.DataFrame(columns=['Bias', 'Token/Word', 'Value', 'TF-IDF'])
        for b in biases:
            tokens = ut_per_class[b].keys()
            words = uw_per_class[b].keys()
            b_rows: List[Dict] = []
            for t in tqdm(tokens):
                tfidf = np.log(1 + ut_per_class[b][t] / count_all_t[b]) * np.log(len(biases)/sum(1 for d in biases if ut_per_class[d][t] > 0)) * 100
                b_rows.append({
                    'Bias': b,
                    'Token/Word': 'Token',
                    'Value': t,
                    'TF-IDF': tfidf,
                })
            b_rows = sorted(b_rows, key=lambda r: r['TF-IDF'], reverse=True)[:self.n]
            df = df.append(b_rows, ignore_index=True)
            b_rows: List[Dict] = []
            for w in tqdm(words):
                tfidf = np.log(1 + uw_per_class[b][w] / count_all_w[b]) * np.log(len(biases)/sum(1 for d in biases if uw_per_class[d][w] > 0)) * 100
                b_rows.append({
                    'Bias': b,
                    'Token/Word': 'Word',
                    'Value': w,
                    'TF-IDF': tfidf,
                })
            b_rows = sorted(b_rows, key=lambda r: r['TF-IDF'], reverse=True)[:self.n]
            df = df.append(b_rows, ignore_index=True)

        self._df = df

    def generate_tables(self):
        biases = sorted(self._df['Bias'].unique().tolist())
        for b in biases:
            for tw in ['Token', 'Word']:
                df = self._df[self._df['Bias'] == b]
                df = df[df['Token/Word'] == tw]
                df = df.drop(columns=['Bias', 'Token/Word'])
                latex_table = tabulate(df.to_dict(orient='records'), headers='keys', tablefmt='latex')
                with open(os.path.join(self.report_dir, self.report_name, f'table_{b}_{tw.lower()}.latex.txt'), 'w') as f:
                    f.write(latex_table)
                grid_table = tabulate(df.to_dict(orient='records'), headers='keys', tablefmt='grid')
                with open(os.path.join(self.report_dir, self.report_name, f'table_{b}_{tw.lower()}.grid.txt'), 'w') as f:
                    f.write(grid_table)

    def generate_graphs(self):
        biases = sorted(self._df['Bias'].unique().tolist())
        for b in biases:
            for tw in ['Token', 'Word']:
                df = self._df[self._df['Bias'] == b]
                df = df[df['Token/Word'] == tw]
                df = df.drop(columns=['Bias', 'Token/Word'])

                figure = plt.gcf()
                figure.set_size_inches(10, 6)
                sns.set_theme('paper')
                g = sns.catplot(y='Value', x='TF-IDF', data=df, kind='bar', palette='coolwarm')
                plt.ylabel(tw)
                plt.title(f'TF-IDF of {tw.lower()}s in document of class {b}')
                g.savefig(os.path.join(self.report_dir, self.report_name, f'tf_idf_{tw.lower()}_{b}.png'), dpi=300)
                plt.clf()
