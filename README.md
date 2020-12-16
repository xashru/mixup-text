# mixup-text
This repository contains implementation of mixup strategy for text classification. The implementation is primarily based on the paper [Augmenting Data with Mixup for Sentence Classification: An Empirical Study
](https://arxiv.org/abs/1905.08941), although there is some difference.

Three variants of mixup are considered for text classification
1. Embedding mixup: Texts are mixed immediately after word embeedding
2. Hidden/Encoder mixup: Mixup is done prior to the last fully connected layer
3.  Sentence mixup: Mixup is done before softmax

#### Results


Some experimental results on TREC, SST-1, IMDB, AG's News and DBPedia datasets. *rand* referes to models initialized randomly. *finetune* is models initialized with pretrained word vector (GloVe or BERT).

| Model                        | TREC  | SST-1 | IMDB  | AG's News | DBPedia |
|------------------------------|-------|-------|-------|-----------|---------|
| CNN-rand                     | 88.58 | 37.00 | 86.74 | 91.07     | 98.03   |
| CNN-rand + embed mixup       | 88.38 | 35.93 | 87.34 | 91.67     | 97.85   |
| CNN-rand + hidden mixup      | 88.78 | 35.24 | 87.06 | 91.49     | 98.34   |
| CNN-rand + sent mixup        | 88.92 | 35.40 | 87.25 | 91.46     | 98.23   |
| CNN-finetune                 | 90.50 | 46.38 | 88.57 | 92.67     | 98.81   |
| CNN-finetune + embed mixup   | 91.62 | 45.81 | 89.13 | 92.78     | 98.55   |
| CNN-finetune + hidden-mixup  | 91.74 | 45.70 | 89.66 | 93.11     | 98.83   |
| CNN-fine-tune + sent mixup   | 91.70 | 46.10 | 89.60 | 93.12     | 98.83   |
| LSTM-finetune                | 89.26 | 44.38 | 86.04 | 92.87     | 98.95   |
| LSTM-finetune + embed mixup  | 89.82 | 44.04 | 85.82 | 92.76     | 98.98   |
| LSTM finetune + hidden mixup | 89.72 | 43.87 | 85.23 | 92.67     | 98.92   |
| LSTM finetune + sent mixup   | 89.70 | 43.86 | 85.02 | 92.65     | 98.87   |
| fastText-finetune            | 86.88 | 43.26 | 88.33 | 91.93     | 97.85   |
| fastText-finetune + mixup    | 86.2  | 43.81 | 88.05 | 91.99     | 97.99   |
| BERT-finetune                | 97.04 | 53.05 | -     | -         | -       |
| BERT-finetune + embed mixup  | 97.20 | 53.12 | -     | -         | -       |
| BERT-finetune + hidden mixup | 96.92 | 53.13 | -     | -         | -       |
| BERT-finetune + sent mixup   | 96.86 | 53.32 | -     | -         | -       |

Results are mean accuracy of 10 runs for all datasets, except for DBPedia where it is average of 3 runs.
Note that for fastText model there is only one variant of mixup as it is a linear model.

#### TO-DO
- [ ] Manifold mixup implementation
- [ ] Result for BERT on IMDB, AG's News and DBPedia datasets