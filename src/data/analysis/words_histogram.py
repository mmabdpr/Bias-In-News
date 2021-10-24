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
class WordsHistogram(PhaseOneAnalysisABC):
    @property
    def required_files(self) -> List[str]:
        return [
            os.path.join(self.data_dir, 'processed', 'tweets_word_broken.json')
        ]

    @property
    def report_name(self) -> str:
        return "words_histogram"

    def __init__(self, data_dir: str, report_dir: str, json_orient: str = 'records'):
        self.data_dir = data_dir
        self.report_dir = report_dir
        self.json_orient = json_orient

        self.check_for_required_files()

        self.data = pd.read_json(os.path.join(self.data_dir, 'processed', 'tweets_word_broken.json'),
                                 orient=self.json_orient)

        # self.data = self.data.sample(frac=0.01).reset_index(drop=True)

        Path(os.path.join(self.report_dir, self.report_dir, self.report_name)).mkdir(parents=True, exist_ok=True)

        self.n = 100
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

        df = pd.DataFrame(columns=['Bias', 'Token/Word', 'Value', 'Count'])
        for b in biases:
            for tw in ['Token', 'Word']:
                tw_d = uw_per_class if tw == 'Word' else ut_per_class
                tws = tw_d[b].keys()
                b_rows: List[Dict] = []
                for t in tqdm(tws):
                    b_rows.append({
                        'Bias': b,
                        'Token/Word': tw,
                        'Value': t,
                        'Count': tw_d[b][t],
                    })
                b_rows = sorted(b_rows, key=lambda r: r['Count'], reverse=True)
                df = df.append(b_rows, ignore_index=True)

        self._df = df

    def generate_tables(self):
        biases = sorted(self._df['Bias'].unique().tolist())
        for b in biases:
            for tw in ['Token', 'Word']:
                df = self._df[self._df['Bias'] == b]
                df = df[df['Token/Word'] == tw]
                df = df.drop(columns=['Bias', 'Token/Word'])
                h, e = np.histogram(np.log(df['Count'].astype(int).to_numpy()))
                hist_df = pd.DataFrame({
                    'Count': h,
                    'Log Frequency (Bin Edge)': e[:-1]
                })
                latex_table = tabulate(hist_df.to_dict(orient='records'), headers='keys', tablefmt='latex')
                with open(os.path.join(self.report_dir, self.report_name, f'table_{b}_{tw.lower()}.latex.txt'), 'w') as f:
                    f.write(latex_table)
                grid_table = tabulate(hist_df.to_dict(orient='records'), headers='keys', tablefmt='grid')
                with open(os.path.join(self.report_dir, self.report_name, f'table_{b}_{tw.lower()}.grid.txt'), 'w') as f:
                    f.write(grid_table)

    def generate_graphs(self):
        biases = sorted(self._df['Bias'].unique().tolist())
        for b in biases:
            for tw in ['Token', 'Word']:
                df = self._df[self._df['Bias'] == b]
                df = df[df['Token/Word'] == tw]
                df = df.drop(columns=['Bias', 'Token/Word'])
                df['Frequency'] = df['Count'].astype(int)
                figure = plt.gcf()
                figure.set_size_inches(10, 6)
                sns.set_theme('paper')
                g = sns.histplot(x=np.log(df['Frequency'].to_numpy()), palette='coolwarm', kde=True, bins=10)
                plt.title(f'histogram for log of {tw.lower()} frequencies in document of class {b}')
                plt.xlabel('Log of Frequency')
                g.figure.savefig(os.path.join(self.report_dir, self.report_name, f'hist_{tw.lower()}_{b}.png'), dpi=300)
                plt.clf()
