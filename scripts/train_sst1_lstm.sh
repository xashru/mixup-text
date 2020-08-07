# Random
python train_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-none

python train_mixup_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-embed --method=embed

python train_mixup_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-sent --method=sent

python train_mixup_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-dense --method=dense

# nonstatic
python train_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-nonstatic-none --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-nonstatic-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-nonstatic-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-nonstatic-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=True

# static
python train_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-static-none --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-static-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-static-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextLSTM --task=sst1 --save-path='../drive/My Drive/research/mixup-text/logs-sst1' --name=sst1-textLSTM-static-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=False
