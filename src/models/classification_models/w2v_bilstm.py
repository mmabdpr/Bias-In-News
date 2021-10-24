import datetime
import json
import logging
import os
import time
from pathlib import Path

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from spacy.lang.en import English
from torch import nn
from torch.utils.data import DataLoader
import gc

from src.data.dataset_utils import generate_small_dataset, split_dataset

project_dir = Path(__file__).resolve().parents[3]
data_dir = os.path.join(project_dir, "data")
models_dir = os.path.join(project_dir, "models")
json_orient = 'records'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def flat_accuracy(preds, labels):
    pred_flat = preds.flatten()

    def logit_to_class(logit):
        if logit < -1.5:
            return -2.
        if -1.5 <= logit < -0.5:
            return -1.
        if -0.5 <= logit < 0.5:
            return 0.
        if 0.5 <= logit < 1.5:
            return 1.
        if 1.5 <= logit:
            return 2.

    pred_flat = np.array([logit_to_class(x) for x in pred_flat])
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


class W2VBiLstmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, seq_len=64):
        self.encodings = encodings
        self.labels = labels
        self.seq_len = seq_len

    def __getitem__(self, idx):
        item = dict()
        ln = len(self.encodings[idx])
        if ln >= self.seq_len:
            item['input_ids'] = torch.LongTensor(self.encodings[idx][:self.seq_len])
        else:
            pad_count = self.seq_len - ln
            item['input_ids'] = F.pad(torch.LongTensor(self.encodings[idx]), (0, pad_count))
        item['labels'] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


class W2VBiLstmModel(nn.Module):
    def __init__(self, trainable_embedding=False):
        super(W2VBiLstmModel, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors))
        self.embedding.weight.requires_grad = trainable_embedding

        self.lstm = nn.LSTM(300, 256, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.lstm2 = nn.LSTM(512, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.linear1 = nn.Linear(128 * 2, 1)

    def forward(self, ids):
        x = self.embedding(ids)

        x, (h, c) = self.lstm(x)
        x, (h, c) = self.lstm2(x)
        hidden = torch.cat((x[:, -1, :128], x[:, 0, 128:]), dim=-1)

        x = self.linear1(hidden.view(-1, 128 * 2))
        x = torch.mul(torch.tanh(x), 2)

        return x


class W2VBiLstmTrainer:
    def __init__(self, debug=False):
        self.debug = debug
        self.tokenizer = None
        self.train_dataset = None
        self.validate_dataset = None
        self.test_dataset = None
        self.model = None
        self.optim = None
        self.batch_size = 32
        self.criterion = torch.nn.MSELoss()
        self.output_dir = Path(models_dir) / 'w2v_bilstm'
        self.best_val_accuracy = -1

    def load_dataset(self):
        nlp = English()

        def tokenizer(s):
            return list(pd.Series([wv.key_to_index.get(t.text) for t in nlp.tokenizer(s)], dtype=float).dropna())

        self.tokenizer = tokenizer

        path = os.path.join(data_dir, 'processed', 'tweets_small.json' if self.debug else 'tweets.json')

        tweets = pd.read_json(path, orient=json_orient).sample(frac=1., random_state=42)
        train, validate, test = split_dataset(tweets)
        del tweets

        train_encodings, train_labels = train['tweet'].apply(self.tokenizer).to_list(), train['bias'].to_list()
        validate_encodings, validate_labels = validate['tweet'].apply(self.tokenizer).to_list(), validate[
            'bias'].to_list()
        test_encodings, test_labels = test['tweet'].apply(self.tokenizer).to_list(), test['bias'].to_list()

        self.train_dataset = W2VBiLstmDataset(train_encodings, train_labels)
        self.validate_dataset = W2VBiLstmDataset(validate_encodings, validate_labels)
        self.test_dataset = W2VBiLstmDataset(test_encodings, test_labels)

    def make_model(self, fine_tune_embedding=False):
        self.model = W2VBiLstmModel(trainable_embedding=fine_tune_embedding).to(device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

    def train_model(self, overwrite_checkpoint=False, reset_eval_acc=False):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite_checkpoint:
            logging.info("Trying to load the best model..")
            self.load_best_model()

        for epoch in range(200):
            logging.info(f"epoch {epoch + 1}")
            total_train_loss = 0
            total_train_accuracy = 0
            t0 = time.time()
            self.model.train()
            for step, batch in enumerate(train_loader):
                if (step % 10 == 0 and not step == 0) or step == len(train_loader) - 1:
                    elapsed = format_time(time.time() - t0)
                    logging.info(
                        '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step + 1, len(train_loader), elapsed))
                self.optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(torch.float).to(device)
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                total_train_loss += loss.item()

                outputs = outputs.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()

                total_train_accuracy += flat_accuracy(outputs, label_ids)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()

            avg_train_loss = total_train_loss / len(train_loader)
            training_time = format_time(time.time() - t0)

            avg_train_accuracy = total_train_accuracy / len(train_loader)
            logging.info("  Train Accuracy: {0:.2f}".format(avg_train_accuracy))

            logging.info("  Average training loss: {0:.2f}".format(avg_train_loss))
            logging.info("  Training epoch took: {:}".format(training_time))

            logging.info("Running Validation...")

            t0 = time.time()
            self.model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0

            validation_dataloader = DataLoader(self.validate_dataset, batch_size=self.batch_size, shuffle=True)

            for batch in validation_dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(torch.float).to(device)

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    loss = self.criterion(outputs, labels)

                    total_eval_loss += loss.item()

                    outputs = outputs.detach().cpu().numpy()
                    label_ids = labels.to('cpu').numpy()

                    total_eval_accuracy += flat_accuracy(outputs, label_ids)

            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            logging.info("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))

            avg_val_loss = total_eval_loss / len(validation_dataloader)

            validation_time = format_time(time.time() - t0)

            logging.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
            logging.info("  Validation took: {:}\n".format(validation_time))

            if self.best_val_accuracy <= avg_val_accuracy:
                logging.info("Improved validation accuracy. Saving the model..")
                self.best_val_accuracy = avg_val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'loss': self.criterion,
                    'val_acc': self.best_val_accuracy
                }, self.output_dir / 'best.pth')
                results = {
                    'epoch': epoch,
                    'train loss': avg_train_loss,
                    'train_acc': avg_train_accuracy,
                    'val_loss': avg_val_loss,
                    'val_acc': avg_val_accuracy
                }
                with open(self.output_dir / 'best_results.json', 'w') as f:
                    json.dump(results, f)

    def evaluate_model(self):
        self.load_best_model()

        t0 = time.time()
        self.model.eval()

        total_test_accuracy = 0
        total_test_loss = 0

        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(torch.float).to(device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)

                total_test_loss += loss.item()

                outputs = outputs.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()

                total_test_accuracy += flat_accuracy(outputs, label_ids)

        avg_val_accuracy = total_test_accuracy / len(test_dataloader)
        logging.info("  Testing Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_test_loss / len(test_dataloader)

        validation_time = format_time(time.time() - t0)

        logging.info("  Testing Loss: {0:.2f}".format(avg_val_loss))
        logging.info("  Testing took: {:}".format(validation_time))

        results = {
            'test loss': avg_val_loss,
            'test acc': avg_val_accuracy
        }

        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f)

    def load_best_model(self):
        if (self.output_dir / 'best.pth').is_file():
            logging.info('Best model found. Loading parameters..')

            gc.collect()
            torch.cuda.empty_cache()

            checkpoint = torch.load(self.output_dir / 'best.pth', map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            self.best_val_accuracy = checkpoint['val_acc']
            logging.info(f'Model was trained for {epoch} epochs. Best val acc: {self.best_val_accuracy}')
            self.criterion = checkpoint['loss']

            del checkpoint

            gc.collect()
            torch.cuda.empty_cache()
        else:
            logging.info('Could not find the best model.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    generate_small_dataset(n_per_class=1000)

    wv = api.load('word2vec-google-news-300')

    trainer = W2VBiLstmTrainer(debug=True)
    trainer.load_dataset()
    trainer.make_model(fine_tune_embedding=False)
    trainer.train_model(overwrite_checkpoint=True, reset_eval_acc=True)
    trainer.evaluate_model()
