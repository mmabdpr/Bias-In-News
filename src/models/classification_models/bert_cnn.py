import datetime
import gc
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


class BertCnnDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


class BertCnnModel(nn.Module):
    def __init__(self, fine_tune_bert=False):
        super(BertCnnModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = fine_tune_bert

        self.dropout = nn.Dropout(0.25)

        self.embedding_size = 768
        self.seq_len = 512
        self.out_size = 128

        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5

        self.stride = 1

        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, (self.kernel_1,), (self.stride,))
        self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, (self.kernel_2,), (self.stride,))
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, (self.kernel_3,), (self.stride,))
        self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, (self.kernel_4,), (self.stride,))

        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        self.fc = nn.Linear(self.in_features_fc(), 1)

    def in_features_fc(self):
        """Calculates the number of output features after Convolution + Max pooling

        Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        """
        # Calculate size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        # Calculate size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)

        # Calculate size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)

        # Calculate size of convolved/pooled features for convolution_4/max_pooling_4 features
        out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_conv_4 = math.floor(out_conv_4)
        out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_pool_4 = math.floor(out_pool_4)

        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_size

    def forward(self, ids, attention_mask):
        x = self.bert(ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        x = x.last_hidden_state
        x = F.pad(x, (0, 0, 0, self.seq_len - x.shape[1], 0, 0), mode='constant', value=0)
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)

        x2 = self.conv_2(x)
        x2 = torch.relu(x2)
        x2 = self.pool_2(x2)

        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)

        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size()[0], -1)

        out = self.fc(union)

        out = torch.mul(torch.tanh(out), 2)

        return out


# noinspection DuplicatedCode
class BertCnnTrainer:
    def __init__(self, debug=False):
        self.debug = debug
        self.tokenizer = None
        self.train_dataset = None
        self.validate_dataset = None
        self.test_dataset = None
        self.model = None
        self.model = None
        self.optim = None
        self.batch_size = 64
        self.criterion = torch.nn.MSELoss()
        self.output_dir = Path(models_dir) / 'bert_cnn'
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

        self.train_dataset = BertCnnDataset(train_encodings, train_labels)
        self.validate_dataset = BertCnnDataset(validate_encodings, validate_labels)
        self.test_dataset = BertCnnDataset(test_encodings, test_labels)

    def make_model(self, fine_tune_embedding=False):
        self.model = BertCnnModel(fine_tune_embedding).to(device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=0.1)

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


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    generate_small_dataset(n_per_class=1000)

    trainer = BertCnnTrainer(debug=True)
    trainer.load_dataset()
    trainer.make_model(fine_tune_embedding=False)
    trainer.train_model(reset_eval_acc=True)
    trainer.evaluate_model()
