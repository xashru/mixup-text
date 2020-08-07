# Random
python train_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-none

python train_mixup_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-embed --method=embed

python train_mixup_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-sent --method=sent

python train_mixup_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-dense --method=dense

# nonstatic
python train_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-nonstatic-none --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-nonstatic-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-nonstatic-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-nonstatic-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=True

# static
python train_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-static-none --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-static-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-static-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextLSTM --task=imdb --save-path='../drive/My Drive/research/mixup-text/logs-imdb'  --name=imdb-textLSTM-static-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=False
