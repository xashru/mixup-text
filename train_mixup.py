import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

import mixup
import models
from datasets import WordDataset
from utils import get_task_config


def parse_args():
    parser = argparse.ArgumentParser(description='Mixup for text classification')
    parser.add_argument('--task', default='trec', type=str, help='Task name')
    parser.add_argument('--name', default='cnn-text-fine-tune', type=str, help='name of the experiment')
    parser.add_argument('--text-column', default='text', type=str, help='text column name of csv file')
    parser.add_argument('--label-column', default='label', type=str, help='column column name of csv file')
    parser.add_argument('--w2v-file', default=None, type=str, help='word embedding file')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--clip', default=3.0, type=float, help='clip gradient norm')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--decay', default=0., type=float, help='weight decay')
    parser.add_argument('--model', default="TextCNN", type=str, help='model type (default: TextCNN)')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size (default: 128)')
    parser.add_argument('--epoch', default=50, type=int, help='total epochs (default: 200)')
    parser.add_argument('--fine-tune', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to fine-tune embedding or not')
    parser.add_argument('--method', default='embed', type=str, help='which mixing method to use (default: none)')
    parser.add_argument('--alpha', default=1., type=float, help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--save-path', default='out', type=str, help='output log/result directory')
    args = parser.parse_args()
    return args


def mixup_data(x, y, args):
    """Returns mixed inputs, pairs of targets, and lambda"""
    return mixup.MIXUP_METHODS[args.method](x, y, args.alpha)


def mixup_criterion_cross_entropy(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Classification:
    def __init__(self, args):
        self.args = args

        use_cuda = args.cuda and torch.cuda.is_available()

        # for reproducibility
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

        mixup.use_cuda = use_cuda

        self.config = get_task_config(args.task)

        # data loaders
        dataset = WordDataset(self.config.sequence_len, args.batch_size)
        dataset.load_data(self.config.train_file, self.config.test_file, self.config.val_file, args.w2v_file,
                          args.text_column, args.label_column)
        self.train_iterator = dataset.train_iterator
        self.val_iterator = dataset.val_iterator
        self.test_iterator = dataset.test_iterator

        # model
        vocab_size = len(dataset.vocab)
        self.model = models.__dict__[args.model](vocab_size=vocab_size, sequence_len=self.config.sequence_len,
                                                 num_class=self.config.num_class,
                                                 word_embeddings=dataset.word_embeddings, fine_tune=args.fine_tune,
                                                 dropout=args.dropout)
        self.device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
        self.model.to(self.device)

        # logs
        os.makedirs(args.save_path, exist_ok=True)
        self.model_save_path = os.path.join(args.save_path, args.name + '_weights.pt')
        self.log_path = os.path.join(args.save_path, args.name + '_logs.csv')
        print(str(args))
        with open(self.log_path, 'a') as f:
            f.write(str(args) + '\n')
        with open(self.log_path, 'a', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['mode', 'epoch', 'step', 'loss', 'acc'])

        # optimizer
        self.criterion = nn.CrossEntropyLoss()
        if args.task in ['trec', 'sst1']:
            self.optimizer = torch.optim.Adam(self.model.parameters())
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.decay)

        # for early stopping
        self.best_val_acc = 0
        self.early_stop = False
        self.val_patience = 0  # successive iteration when validation acc did not improve

        self.iteration_number = 0

    def test(self, iterator):
        self.model.eval()
        test_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            # for _, batch in tqdm(enumerate(iterator), total=len(iterator), desc='test'):
            for _, batch in enumerate(iterator):
                x = batch.text
                y = batch.label
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                test_loss += loss.item() * y.shape[0]
                total += y.shape[0]
                correct += torch.sum(torch.argmax(y_pred, dim=1) == y).item()

        avg_loss = test_loss / total
        acc = 100.0 * correct / total
        return avg_loss, acc

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        total = 0
        correct = 0
        # for _, batch in tqdm(enumerate(self.train_iterator), total=len(self.train_iterator), desc='train'):
        for _, batch in enumerate(self.train_iterator):
            x = batch.text
            y = batch.label
            x, y = x.to(self.device), y.to(self.device)
            lam = np.random.beta(self.args.alpha, self.args.alpha)
            index = mixup.get_perm(x)
            x1 = x[:, index]
            y1 = y[index]

            if self.args.method == 'embed':
                y_pred = self.model.forward_mix_embed(x, x1, lam)
            elif self.args.method == 'sent':
                y_pred = self.model.forward_mix_sent(x, x1, lam)
            elif self.args.method == 'dense':
                y_pred = self.model.forward_mix_encoder(x, x1, lam)
            else:
                raise ValueError('invalid method name')

            loss = mixup_criterion_cross_entropy(self.criterion, y_pred, y, y1, lam)
            train_loss += loss.item() * y.shape[0]
            total += y.shape[0]
            _, predicted = torch.max(y_pred.data, 1)
            correct += ((lam * predicted.eq(y.data).cpu().sum().float()
                         + (1 - lam) * predicted.eq(y1.data).cpu().sum().float())).item()

            self.optimizer.zero_grad()
            loss.backward()
            if self.args.task in ['trec', 'sst1']:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()

            # eval
            self.iteration_number += 1
            if self.iteration_number % self.config.eval_interval == 0:
                avg_loss = train_loss / total
                acc = 100.0 * correct / total
                # print('Train loss: {}, Train acc: {}'.format(avg_loss, acc))
                train_loss = 0
                total = 0
                correct = 0

                val_loss, val_acc = self.test(iterator=self.val_iterator)
                # print('Val loss: {}, Val acc: {}'.format(val_loss, val_acc))
                if val_acc > self.best_val_acc:
                    torch.save(self.model.state_dict(), self.model_save_path)
                    self.best_val_acc = val_acc
                    self.val_patience = 0
                else:
                    self.val_patience += 1
                    if self.val_patience == self.config.patience:
                        self.early_stop = True
                        return
                with open(self.log_path, 'a', newline='') as out:
                    writer = csv.writer(out)
                    writer.writerow(['train', epoch, self.iteration_number, avg_loss, acc])
                    writer.writerow(['val', epoch, self.iteration_number, val_loss, val_acc])
                self.model.train()

    def run(self):
        for epoch in range(self.args.epoch):
            print('------------------------------------- Epoch {} -------------------------------------'.format(epoch))
            self.train(epoch)
            if self.early_stop:
                break
        print('Training complete!')
        print('Best Validation Acc: ', self.best_val_acc)

        self.model.load_state_dict(torch.load(self.model_save_path))
        train_loss, train_acc = self.test(self.train_iterator)
        val_loss, val_acc = self.test(self.val_iterator)
        test_loss, test_acc = self.test(self.test_iterator)

        with open(self.log_path, 'a', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['train', -1, -1, train_loss, train_acc])
            writer.writerow(['val', -1, -1, val_loss, val_acc])
            writer.writerow(['test', -1, -1, test_loss, test_acc])

        print('Train loss: {}, Train acc: {}'.format(train_loss, train_acc))
        print('Val loss: {}, Val acc: {}'.format(val_loss, val_acc))
        print('Test loss: {}, Test acc: {}'.format(test_loss, test_acc))

        return val_acc, test_acc


if __name__ == '__main__':
    cls = Classification(parse_args())
    cls.run()
