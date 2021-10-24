import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from dotenv import load_dotenv, find_dotenv
from tabulate import tabulate
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.data.dataset_utils import get_class_labels, create_corpus

project_dir = Path(__file__).resolve().parents[3]
data_dir = os.path.join(project_dir, "data")
models_dir = os.path.join(project_dir, "models")
report_dir = os.path.join(project_dir, 'docs', 'phase_2_report', 'figs')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class GPT2CLMExperiment(ABC):
    @abstractmethod
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, report_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.report_name = report_name

        self.samples = [
            'It is true that',
            'Democrats are',
            'Republicans are',
            'There is an argument about abortion.',
            'Protestors want to ',
            'There a huge benefit in',
            'In this country immigrants should ',
            'We have to think twice about going after ',
            'Is there any evidence that ',
            'Government should do something about global warming',
            'Global warming is a ',
            'In this country guns should be ',
            'Chinese government has a plan to',
            'If we want to make something legal it should be',
            'If we want to make something illegal it should be',
            'Socialism has the power to',
            'Fake news ',
            'Voters do not have the ability to ',
            'Democrats want the voters to ',
        ]

        self.results = {
            'Input': [],
            'GPT2 Output': []
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


class GPT2CLMWrapper:
    def __init__(self):
        self.model_name_or_path = 'gpt2'

    def fine_tune(self):
        create_corpus(n_per_class=1000)

        classes = get_class_labels()

        logging.info(f"fine-tuning the language model {self.model_name_or_path}")
        for cls in classes:
            run_clm_path = Path(__file__).parents[0]/'run_clm.py'
            path_to_train_file = os.path.join(data_dir, 'processed', 'corpus', f'tweets_corpus_{cls}.txt')
            path_to_validation_file = path_to_train_file
            output_dir = os.path.join(models_dir, f'{self.model_name_or_path}_clm_{cls}')

            if os.path.isdir(output_dir):
                logging.info(f"directory {output_dir} exists. skipping..")
                continue

            os.system(f"""python {run_clm_path} \
                                    --output_dir={output_dir} \
                                    --model_name_or_path=gpt2 \
                                    --do_train \
                                    --train_file={path_to_train_file} \
                                    --do_eval \
                                    --validation_file={path_to_validation_file} \
                                    --num_train_epochs=3 \
                                    --block_size=200
                                """)

            # TODO save perplexity to reports

    def run_experiments(self):
        experiments = [
            E1,
            E2,
            E3,
            E4,
            E5,
            E6,
            E7,
            E8,
        ]

        for cls in get_class_labels():
            output_dir = os.path.join(models_dir, f'{self.model_name_or_path}_clm_{cls}')
            model, tokenizer = self._load_from_pretrained(output_dir)

            for exp_type in experiments:
                logging.info(f'experiment: {exp_type.__name__} ; class: {cls}')
                exp = exp_type(model, tokenizer, report_name=f'gpt2_clm_class_{cls}')
                logging.info(f'experiment description: {exp.description}')
                exp.run()
                exp.save_results()

    @staticmethod
    def _load_from_pretrained(name_or_path='gpt2'):
        tokenizer = GPT2Tokenizer.from_pretrained(name_or_path)
        # add the EOS token as PAD token to avoid warnings
        model = GPT2LMHeadModel.from_pretrained(name_or_path, pad_token_id=tokenizer.eos_token_id).to(device)
        return model, tokenizer


class E1(GPT2CLMExperiment):
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, report_name: str):
        super().__init__(model, tokenizer, report_name)

    def run(self):
        for sample in self.samples:
            input_ids = torch.LongTensor(self.tokenizer.encode(sample, return_tensors='pt')).to(device)

            beam_output = self.model.generate(
                input_ids,
                max_length=30,
                num_beams=5,
                early_stopping=True,
            )

            res = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
            logging.info(res)

            self.results['Input'].append(sample)
            self.results['GPT2 Output'].append(res)

    @property
    def description(self) -> str:
        return "activate beam search and early_stopping"


class E2(GPT2CLMExperiment):
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, report_name: str):
        super().__init__(model, tokenizer, report_name)

    def run(self):
        for sample in self.samples:
            input_ids = torch.LongTensor(self.tokenizer.encode(sample, return_tensors='pt')).to(device)

            beam_output = self.model.generate(
                input_ids,
                max_length=30,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

            res = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
            logging.info(res)

            self.results['Input'].append(sample)
            self.results['GPT2 Output'].append(res)

    @property
    def description(self) -> str:
        return "set no_repeat_ngram_size to 2"


class E3(GPT2CLMExperiment):
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, report_name: str):
        super().__init__(model, tokenizer, report_name)

    def run(self):
        for sample in self.samples:
            input_ids = torch.LongTensor(self.tokenizer.encode(sample, return_tensors='pt')).to(device)

            beam_outputs = self.model.generate(
                input_ids,
                max_length=30,
                num_beams=5,
                no_repeat_ngram_size=2,
                num_return_sequences=5,
                early_stopping=True
            )

            for i, beam_output in enumerate(beam_outputs):
                res = self.tokenizer.decode(beam_output, skip_special_tokens=True)
                logging.info(f'result {i + 1}: {res}')

                self.results['Input'].append(sample)
                self.results['GPT2 Output'].append(res)

    @property
    def description(self) -> str:
        return "set return_num_sequences > 1"


class E4(GPT2CLMExperiment):
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, report_name: str):
        super().__init__(model, tokenizer, report_name)

    def run(self):
        for sample in self.samples:
            input_ids = torch.LongTensor(self.tokenizer.encode(sample, return_tensors='pt')).to(device)

            sample_output = self.model.generate(
                input_ids,
                do_sample=True,
                max_length=50,
                top_k=0
            )

            res = self.tokenizer.decode(sample_output[0], skip_special_tokens=True)
            logging.info(res)

            self.results['Input'].append(sample)
            self.results['GPT2 Output'].append(res)

    @property
    def description(self) -> str:
        return "activate sampling and deactivate top_k by setting top_k sampling to 0"


class E5(GPT2CLMExperiment):
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, report_name: str):
        super().__init__(model, tokenizer, report_name)

    def run(self):
        for sample in self.samples:
            input_ids = torch.LongTensor(self.tokenizer.encode(sample, return_tensors='pt')).to(device)

            sample_output = self.model.generate(
                input_ids,
                do_sample=True,
                max_length=50,
                top_k=0,
                temperature=0.7
            )

            res = self.tokenizer.decode(sample_output[0], skip_special_tokens=True)
            logging.info(res)

            self.results['Input'].append(sample)
            self.results['GPT2 Output'].append(res)

    @property
    def description(self) -> str:
        return "use temperature to decrease the sensitivity to low probability candidates"


class E6(GPT2CLMExperiment):
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, report_name: str):
        super().__init__(model, tokenizer, report_name)

    def run(self):
        for sample in self.samples:
            input_ids = torch.LongTensor(self.tokenizer.encode(sample, return_tensors='pt')).to(device)

            sample_output = self.model.generate(
                input_ids,
                do_sample=True,
                max_length=50,
                top_k=50
            )

            res = self.tokenizer.decode(sample_output[0], skip_special_tokens=True)
            logging.info(res)

            self.results['Input'].append(sample)
            self.results['GPT2 Output'].append(res)

    @property
    def description(self) -> str:
        return "set top_k to 50"


class E7(GPT2CLMExperiment):
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, report_name: str):
        super().__init__(model, tokenizer, report_name)

    def run(self):
        for sample in self.samples:
            input_ids = torch.LongTensor(self.tokenizer.encode(sample, return_tensors='pt')).to(device)

            sample_output = self.model.generate(
                input_ids,
                do_sample=True,
                max_length=50,
                top_p=0.92,
                top_k=0
            )

            res = self.tokenizer.decode(sample_output[0], skip_special_tokens=True)
            logging.info(res)

            self.results['Input'].append(sample)
            self.results['GPT2 Output'].append(res)

    @property
    def description(self) -> str:
        return "deactivate top_k sampling and sample only from 92% most likely words"


class E8(GPT2CLMExperiment):
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, report_name: str):
        super().__init__(model, tokenizer, report_name)

    def run(self):
        for sample in self.samples:
            input_ids = torch.LongTensor(self.tokenizer.encode(sample, return_tensors='pt')).to(device)

            sample_outputs = self.model.generate(
                input_ids,
                do_sample=True,
                max_length=50,
                top_k=50,
                top_p=0.95,
                num_return_sequences=3
            )

            for i, sample_output in enumerate(sample_outputs):
                res = self.tokenizer.decode(sample_output, skip_special_tokens=True)

                logging.info(f'result {i+1}: {res}')

                self.results['Input'].append(sample)
                self.results['GPT2 Output'].append(res)

    @property
    def description(self) -> str:
        return "set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3"


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    gpt2 = GPT2CLMWrapper()
    gpt2.fine_tune()
    gpt2.run_experiments()
