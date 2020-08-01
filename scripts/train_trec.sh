# input or none
python train.py --sequence-len=20 --model=TextCNN --method=none --seed=1 --alpha=1 --save-path='../drive/My Drive/research/mixup-text/logs/1' --train-file=data/trec/train.csv  --val-file=data/trec/val.csv --test-file=data/trec/test.csv --label-column=label --num-class=6

# embedding
python train_embed_mixup.py --sequence-len=20 --model=TextCNN --method=mixup --seed=1 --alpha=1 --save-path='../drive/My Drive/research/mixup-text/logs/2' --train-file=data/trec/train.csv  --val-file=data/trec/val.csv --test-file=data/trec/test.csv --label-column=label --num-class=6