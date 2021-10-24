import datetime
import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizerFast

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


class BertBiLstmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


class BertBiLstmModel(nn.Module):
    def __init__(self, fine_tune_bert=False):
        super(BertBiLstmModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = fine_tune_bert

        # self.lstm = nn.LSTM(768, 512, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        # self.linear1 = nn.Linear(512 * 2, 512)
        # self.bn1 = nn.BatchNorm1d(num_features=512)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(num_features=256)
        # self.linear3 = nn.Linear(256, 128)
        # self.bn3 = nn.BatchNorm1d(num_features=128)
        # self.linear4 = nn.Linear(128, 64)
        # self.bn4 = nn.BatchNorm1d(num_features=64)
        # self.linear5 = nn.Linear(64, 1)

        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.linear1 = nn.Linear(256 * 2, 1)
        # self.bn1 = nn.BatchNorm1d(num_features=512)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(num_features=256)
        # self.linear3 = nn.Linear(256, 128)
        # self.bn3 = nn.BatchNorm1d(num_features=128)
        # self.linear4 = nn.Linear(128, 64)
        # self.bn4 = nn.BatchNorm1d(num_features=64)
        # self.linear5 = nn.Linear(64, 1)

    def forward(self, ids, attention_mask):
        x = self.bert(ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)

        lstm_output, (_, _) = self.lstm(x.last_hidden_state)
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)

        x = self.linear1(hidden.view(-1, 256 * 2))
        # x = self.bn1(x)
        # x = F.relu(x)
        # x = F.dropout(x, 0.3, self.training)
        #
        # x = self.linear2(x)
        # x = self.bn2(x)
        # x = F.relu(x)
        # x = F.dropout(x, 0.25, self.training)
        #
        # x = self.linear3(x)
        # x = self.bn3(x)
        # x = F.relu(x)
        # x = F.dropout(x, 0.2, self.training)
        #
        # x = self.linear4(x)
        # x = self.bn4(x)
        # x = F.relu(x)
        # x = F.dropout(x, 0.15, self.training)
        #
        # x = self.linear5(x)

        x = torch.mul(torch.tanh(x), 2)

        return x


# noinspection DuplicatedCode
class BertBiLstmTrainer:
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
        self.output_dir = Path(models_dir) / 'bert_bilstm'
        self.best_val_accuracy = -1

    def load_dataset(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        path = os.path.join(data_dir, 'processed', 'tweets_small.json' if self.debug else 'tweets.json')

        tweets = pd.read_json(path, orient=json_orient).sample(frac=1., random_state=42)
        train, validate, test = split_dataset(tweets)
        del tweets

        train_texts, train_labels = train['tweet'].to_list(), train['bias'].to_list()
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        del train_texts

        validate_texts, validate_labels = validate['tweet'].to_list(), validate['bias'].to_list()
        validate_encodings = self.tokenizer(validate_texts, truncation=True, padding=True)
        del validate_texts

        test_texts, test_labels = test['tweet'].to_list(), test['bias'].to_list()
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True)
        del test_texts

        self.train_dataset = BertBiLstmDataset(train_encodings, train_labels)
        self.validate_dataset = BertBiLstmDataset(validate_encodings, validate_labels)
        self.test_dataset = BertBiLstmDataset(test_encodings, test_labels)

    def make_model(self, fine_tune_embedding=False):
        self.model = BertBiLstmModel(fine_tune_embedding).to(device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

    def train_model(self, overwrite_checkpoint=False, reset_eval_acc=False):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite_checkpoint:
            logging.info("Trying to load the best model..")
            self.load_best_model()

        if reset_eval_acc:
            self.best_val_accuracy = -1

        for epoch in range(100):
            logging.info(f"epoch {epoch + 1}")
            total_train_loss = 0
            total_train_accuracy = 0
            t0 = time.time()
            self.model.train()
            for step, batch in enumerate(train_loader):
                if (step % 10 == 0 and not step == 0) or step == len(train_loader) - 1:
                    elapsed = format_time(time.time() - t0)
                    logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step + 1, len(train_loader), elapsed))
                self.optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(torch.float).to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
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
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(torch.float).to(device)

                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
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
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(torch.float).to(device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
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

    trainer = BertBiLstmTrainer(debug=True)
    trainer.load_dataset()
    trainer.make_model(fine_tune_embedding=False)
    trainer.train_model()
    trainer.evaluate_model()
