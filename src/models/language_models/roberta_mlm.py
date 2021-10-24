import logging
import os
from pathlib import Path

import torch
from tabulate import tabulate
from transformers import pipeline
from dotenv import find_dotenv, load_dotenv
from abc import ABC, abstractmethod

from src.data.dataset_utils import create_corpus, get_class_labels

project_dir = Path(__file__).resolve().parents[3]
data_dir = os.path.join(project_dir, "data")
models_dir = os.path.join(project_dir, "models")
report_dir = os.path.join(project_dir, 'docs', 'phase_2_report', 'figs')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class RobertaMLMExperiment(ABC):
    @abstractmethod
    def __init__(self, model_path: str, report_name: str):
        self.model_path = model_path
        self.report_name = report_name

        self.results = {
            'Input': [],
            'Roberta Output': [],
            'Score': []
        }

    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    def save_results(self):
        output_dir = Path(report_dir) / self.report_name
        output_dir.mkdir(parents=True, exist_ok=True)

        latex_table = tabulate(self.results, headers='keys', tablefmt='latex')
        with open(output_dir / f'{self.__class__.__name__}.latex', 'w') as f:
            f.write(latex_table)
        grid_table = tabulate(self.results, headers='keys', tablefmt='grid')
        with open(output_dir / f'{self.__class__.__name__}.grid', 'w') as f:
            f.write(grid_table)


class RobertaMLMWrapper:
    def __init__(self):
        self.model_name_or_path = 'roberta-base'

    def fine_tune(self):
        create_corpus(n_per_class=1000)

        classes = get_class_labels()

        logging.info(f"fine-tuning the language model {self.model_name_or_path}")
        for cls in classes:
            run_mlm_path = Path(__file__).parents[0] / 'run_mlm.py'
            path_to_train_file = os.path.join(data_dir, 'processed', 'corpus', f'tweets_corpus_{cls}.txt')
            path_to_validation_file = path_to_train_file
            output_dir = os.path.join(models_dir, f'{self.model_name_or_path}_mlm_{cls}')

            if os.path.isdir(output_dir):
                logging.info(f"directory {output_dir} exists. skipping..")
                continue

            os.system(f"""python {run_mlm_path} \
                                --model_name_or_path {self.model_name_or_path} \
                                --train_file {path_to_train_file} \
                                --validation_file {path_to_validation_file} \
                                --do_train \
                                --do_eval \
                                --line_by_line \
                                --output_dir {output_dir}
                            """)

            # TODO save perplexity to reports

    def run_experiments(self):
        experiments = [
            E1,
        ]

        for cls in get_class_labels():
            model_dir = os.path.join(models_dir, f'{self.model_name_or_path}_mlm_{cls}')

            for exp_type in experiments:
                logging.info(f'experiment: {exp_type.__name__} ; class: {cls}')
                exp = exp_type(model_dir, report_name=f'roberta_mlm_class_{cls}')
                logging.info(f'experiment description: {exp.description}')
                exp.run()
                exp.save_results()


class E1(RobertaMLMExperiment):
    def __init__(self, model_path: str, report_name: str):
        super().__init__(model_path, report_name)

    def run(self):

        mlm = pipeline('fill-mask', model=self.model_path, tokenizer=self.model_path, device=0)
        mlm.top_k = 20
        mask = mlm.tokenizer.mask_token

        phrases = [
            f"Vaccines are {mask}",
            f"Abortion is {mask}",
            f"Immigrants are {mask}",
            f"Conservative senators are {mask}",
            f"Democrat senators are {mask}",
            f"People should vote for {mask}",
            f"Iran nuclear deal is {mask}",
            f"Gun laws {mask} people",
            f"Guns should be {mask}",
            f"Chinese people are {mask}",
            f"{mask} should be illegal",
            f"{mask} should be legal",
            f"The Russian government {mask} the United States",
            f"Capitalism can {mask} people",
            f"Communism can {mask} people",
            f"Socialism can {mask} people",
        ]

        for phrase in phrases:
            res = mlm(phrase)

            logging.info(res)

            for r in res:
                self.results['Input'].append(phrase)
                self.results['Roberta Output'].append(r['token_str'])
                self.results['Score'].append(r['score'])

    @property
    def description(self) -> str:
        return "filling the masked words using the language model roberta-base"


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    roberta = RobertaMLMWrapper()
    roberta.fine_tune()
    roberta.run_experiments()
