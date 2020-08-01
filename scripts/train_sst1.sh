python train_multiple.py --sequence-len=40 --model=TextCNN --method=none --seed=1 --alpha=1 \
--save-path='../drive/My Drive/research/mixup-text/logs/1' --train-file=data/sst1/train.csv  \
--val-file=data/sst1/val.csv --test-file=data/sst1/test.csv --label-column=label --num-class=5 \
--lr=0.001 --batch-size=50 --decay=0
