import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

import models
from datasets import BertDataset
from utils import get_task_config
from models.config import MODELS
from transformers import *


def parse_args():
    parser = argparse.ArgumentParser(description='Mixup for text classification')
    parser.add_argument('--task', default='trec', type=str, help='Task name')
    parser.add_argument('--name', default='cnn-text-fine-tune', type=str, help='name of the experiment')
    parser.add_argument('--text-column', default='text', type=str, help='text column name of csv file')
    parser.add_argument('--label-column', default='label', type=str, help='column column name of csv file')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--decay', default=0., type=float, help='weight decay')
    parser.add_argument('--model', default="bert-base-uncased", type=str, help='pretrained BERT model name')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--batch-size', default=50, type=int, help='batch size (default: 128)')
    parser.add_argument('--epoch', default=20, type=int, help='total epochs (default: 200)')
    parser.add_argument('--fine-tune', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to fine-tune embedding or not')
    parser.add_argument('--save-path', default='out', type=str, help='output log/result directory')
    parser.add_argument('--method', default='none', type=str, help='which mixing method to use (default: none)')
    parser.add_argument('--alpha', default=1., type=float, help='mixup interpolation coefficient (default: 1)')
    args = parser.parse_args()
    return args


class Classification:
    def __init__(self, args):
        self.args = args

        self.use_cuda = args.cuda and torch.cuda.is_available()

        # for reproducibility
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.config = get_task_config(args.task)

        # data loaders
        train_dataset = BertDataset(self.config.train_file, self.config.sequence_len)
        test_dataset = BertDataset(self.config.test_file, self.config.sequence_len)

        if self.config.val_file is None:
            train_samples = int(len(train_dataset) * 0.9)
            val_samples = len(train_dataset) - train_samples
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_samples, val_samples])
        else:
            val_dataset = BertDataset(self.config.val_file, self.config.sequence_len)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # model
        if MODELS[args.model][0] == BertModel:
            self.model = models.TextBERT(pretrained_model=args.model, num_class=self.config.num_class,
                                         fine_tune=args.fine_tune, dropout=args.dropout)

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.decay)

        # for early stopping
        self.best_val_acc = 0
        self.early_stop = False
        self.val_patience = 0  # successive iteration when validation acc did not improve

        self.iteration_number = 0

    def get_perm(self, x):
        """get random permutation"""
        batch_size = x.size()[0]
        if self.use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        return index

    def mixup_criterion_cross_entropy(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def test(self, loader):
        self.model.eval()
        test_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for x, att, y in loader:
                x, y, att = x.to(self.device), y.to(self.device), att.to(self.device)
                y_pred = self.model(x, att)
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
        for x, att, y in self.train_loader:
            x, y, att = x.to(self.device), y.to(self.device), att.to(self.device)
            y_pred = self.model(x, att)
            loss = self.criterion(y_pred, y)
            train_loss += loss.item() * y.shape[0]
            total += y.shape[0]
            correct += torch.sum(torch.argmax(y_pred, dim=1) == y).item()

            self.optimizer.zero_grad()
            loss.backward()
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

                val_loss, val_acc = self.test(self.val_loader)
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

    def train_mixup(self, epoch):
        self.model.train()
        train_loss = 0
        total = 0
        correct = 0
        for x, att, y in self.train_loader:
            x, y, att = x.to(self.device), y.to(self.device), att.to(self.device)
            lam = np.random.beta(self.args.alpha, self.args.alpha)
            index = self.get_perm(x)
            x1 = x[index]
            y1 = y[index]
            att1 = att[index]

            if self.args.method == 'embed':
                y_pred = self.model.forward_mix_embed(x, att, x1, att1, lam)
            elif self.args.method == 'sent':
                y_pred = self.model.forward_mix_sent(x, att, x1, att1, lam)
            elif self.args.method == 'encoder':
                y_pred = self.model.forward_mix_encoder(x, att, x1, att1, lam)
            else:
                raise ValueError('invalid method name')

            loss = self.mixup_criterion_cross_entropy(y_pred, y, y1, lam)
            train_loss += loss.item() * y.shape[0]
            total += y.shape[0]
            _, predicted = torch.max(y_pred.data, 1)
            correct += ((lam * predicted.eq(y.data).cpu().sum().float()
                         + (1 - lam) * predicted.eq(y1.data).cpu().sum().float())).item()

            self.optimizer.zero_grad()
            loss.backward()
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

                val_loss, val_acc = self.test(self.val_loader)
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
            if self.args.method == 'none':
                self.train(epoch)
            else:
                self.train_mixup(epoch)
            if self.early_stop:
                break
        print('Training complete!')
        print('Best Validation Acc: ', self.best_val_acc)

        self.model.load_state_dict(torch.load(self.model_save_path))
        train_loss, train_acc = self.test(self.train_loader)
        val_loss, val_acc = self.test(self.val_loader)
        test_loss, test_acc = self.test(self.test_loader)

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
    args = parse_args()
    num_runs = args.num_runs

    test_acc = []
    val_acc = []

    for i in range(num_runs):
        cls = Classification(args)
        val, test = cls.run()
        val_acc.append(val)
        test_acc.append(test)
        args.seed += 1

    with open(os.path.join(args.save_path, args.name + '_result.txt', 'a')) as f:
        f.write(str(args))
        f.write('val acc:' + str(val_acc) + '\n')
        f.write('test acc:' + str(test_acc) + '\n')
        f.write('mean val acc:' + str(np.mean(val_acc)) + '\n')
        f.write('std val acc:' + str(np.std(val_acc, ddof=1)) + '\n')
        f.write('mean test acc:' + str(np.mean(test_acc)) + '\n')
        f.write('std test acc:' + str(np.std(test_acc, ddof=1)) + '\n\n\n')
