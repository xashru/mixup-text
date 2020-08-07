# Random
python train_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-none

python train_mixup_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-embed --method=embed

python train_mixup_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-sent --method=sent

python train_mixup_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-dense --method=dense

# nonstatic
python train_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-nonstatic-none --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-nonstatic-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-nonstatic-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-nonstatic-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=True

# static
python train_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-static-none --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-static-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-static-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextLSTM --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-textLSTM-static-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=False
