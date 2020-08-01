# Random
python train_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0.5 --name=ag-textcnn-none

python train_mixup_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0 --name=ag-textcnn-embed --method=embed

python train_mixup_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0 --name=ag-textcnn-sent --method=sent

python train_mixup_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0 --name=ag-textcnn-dense --method=dense

# nonstatic
python train_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0.5 --name=ag-textcnn-nonstatic-none --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0 --name=ag-textcnn-nonstatic-embed --w2v-file=data/glove.pickle --fine-tune=True --method=embed

python train_mixup_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0 --name=ag-textcnn-nonstatic-sent --w2v-file=data/glove.pickle --fine-tune=True --method=sent

python train_mixup_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0 --name=ag-textcnn-nonstatic-dense --w2v-file=data/glove.pickle --fine-tune=True --method=dense


# static
python train_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0.5 --name=ag-textcnn-static-none --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0 --name=ag-textcnn-static-embed --w2v-file=data/glove.pickle --fine-tune=False --method=embed

python train_mixup_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0 --name=ag-textcnn-static-sent --w2v-file=data/glove.pickle --fine-tune=False --method=sent

python train_mixup_multiple.py --sequence-len=30 --model=TextCNN --seed=1 --save-path='logs' --train-file=data/trec/train.csv --test-file=data/trec/test.csv --label-column=label --num-class=4 --lr=0.001 --batch-size=50 --eval-interval=20 --patience=30 --dropout=0 --name=ag-textcnn-static-dense --w2v-file=data/glove.pickle --fine-tune=False --method=dense
