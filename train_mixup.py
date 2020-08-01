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


def parse_args():
    parser = argparse.ArgumentParser(description='Mixup for text classification')
    parser.add_argument('--name', default='cnn-text-fine-tune', type=str, help='name of the experiment')
    parser.add_argument('--train-file', default='data/ag_news.train', type=str, help='Train csv file path')
    parser.add_argument('--test-file', default='data/ag_news.test', type=str, help='Test csv file path')
    parser.add_argument('--text-column', default='text', type=str, help='text column name of csv file')
    parser.add_argument('--label-column', default='label', type=str, help='column column name of csv file')
    parser.add_argument('--val-file', default=None, type=str, help='Test csv file path')
    parser.add_argument('--w2v-file', default=None, type=str, help='word embedding file')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--decay', default=0., type=float, help='weight decay')
    parser.add_argument('--model', default="TextCNN", type=str, help='model type (default: TextCNN)')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size (default: 128)')
    parser.add_argument('--sequence-len', default=64, type=int, help='maximum sequence length')
    parser.add_argument('--step-size', default=64, type=int, help='step interval between evaluation')
    parser.add_argument('--num-class', default=4, type=int, help='number of classes (default 4)')
    parser.add_argument('--epoch', default=50, type=int, help='total epochs (default: 200)')
    parser.add_argument('--fine-tune', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to fine-tune embedding or not')
    parser.add_argument('--method', default='mixup', type=str,  help='which mixing method to use (default: none)')
    parser.add_argument('--alpha', default=1., type=float, help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--save-path', default='out', type=str, help='output log/result directory')
    args = parser.parse_args()
    return args


def mixup_criterion_cross_entropy(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, train_iterator, device, args, criterion, optimizer):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for _, batch in tqdm(enumerate(train_iterator), total=len(train_iterator), desc='train'):
        x = batch.text
        y = batch.label
        x, y = x.to(device), y.to(device)
        lam = np.random.beta(args.alpha, args.alpha)
        index = mixup.get_perm(x)
        x1 = x[:, index]
        y1 = y[index]
        y_pred = model.forward_mix_embed(x, x1, lam, args.method)
        loss = mixup_criterion_cross_entropy(criterion, y_pred, y, y1, lam)
        train_loss += loss.item() * y.shape[0]
        total += y.shape[0]
        _, predicted = torch.max(y_pred.data, 1)
        correct += ((lam * predicted.eq(y.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(y1.data).cpu().sum().float())).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = train_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def test(model, iterator, device, criterion):
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(iterator), total=len(iterator), desc='test'):
            x = batch.text
            y = batch.label
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            test_loss += loss.item() * y.shape[0]
            total += y.shape[0]
            correct += torch.sum(torch.argmax(y_pred, dim=1) == y).item()

    avg_loss = test_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def main(args):
    # prepare
    use_cuda = args.cuda and torch.cuda.is_available()

    # for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    mixup.use_cuda = use_cuda

    dataset = WordDataset(args.sequence_len, args.batch_size)
    dataset.load_data(args.train_file, args.test_file, args.val_file, args.w2v_file, args.text_column,
                      args.label_column)
    train_iterator, val_iterator, test_iterator = dataset.train_iterator, dataset.val_iterator, dataset.test_iterator

    vocab_size = len(dataset.vocab)
    model = models.__dict__[args.model](vocab_size=vocab_size, sequence_len=args.sequence_len, num_class=args.num_class,
                                        word_embeddings=dataset.word_embeddings, fine_tune=args.fine_tune)

    os.makedirs(args.save_path, exist_ok=True)
    model_save_path = os.path.join(args.save_path, args.name + '_weights.pt')
    log_path = os.path.join(args.save_path, args.name + '_logs.csv')
    print(str(args))
    with open(log_path, 'a') as f:
        f.write(str(args) + '\n')

    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    with open(log_path, 'a', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['mode', 'epoch', 'loss', 'acc'])

    best_val_acc = 0

    for epoch in range(args.epoch):
        print('--------------------------------------- Epoch {} ---------------------------------------'.format(epoch))
        train_loss, train_acc = train(model=model, train_iterator=train_iterator, device=device, args=args,
                                      criterion=criterion, optimizer=optimizer)
        print('Train loss: {}, Train acc: {}'.format(train_loss, train_acc))
        val_loss, val_acc = test(model=model, iterator=val_iterator, device=device, criterion=criterion)
        print('Val loss: {}, Val acc: {}'.format(val_loss, val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

        with open(log_path, 'a', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['train', epoch, train_loss, train_acc])
            writer.writerow(['val', epoch, val_loss, val_acc])

        lr_scheduler.step(val_loss)

    print('Training complete!')
    print('Best Validation Acc: ', best_val_acc)

    model.load_state_dict(torch.load(model_save_path))
    train_loss, train_acc = test(model=model, iterator=train_iterator, device=device, criterion=criterion)
    val_loss, val_acc = test(model=model, iterator=val_iterator, device=device, criterion=criterion)
    test_loss, test_acc = test(model=model, iterator=test_iterator, device=device, criterion=criterion)

    with open(log_path, 'a', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['train', -1, train_loss, train_acc])
        writer.writerow(['val', -1, val_loss, val_acc])
        writer.writerow(['test', -1, test_loss, test_acc])

    print('Train loss: {}, Train acc: {}'.format(train_loss, train_acc))
    print('Val loss: {}, Val acc: {}'.format(val_loss, val_acc))
    print('Test loss: {}, Test acc: {}'.format(test_loss, test_acc))

    return val_acc, test_acc


if __name__ == "__main__":
    main(parse_args())
