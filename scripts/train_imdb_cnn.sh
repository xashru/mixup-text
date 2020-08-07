# Random
python train_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-none

python train_mixup_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-embed --method=embed

python train_mixup_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-sent --method=sent

python train_mixup_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-dense --method=dense

# nonstatic
python train_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-nonstatic-none --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-nonstatic-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-nonstatic-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-nonstatic-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=True

# static
python train_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-static-none --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-static-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-static-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextCNN --task=imdb --save-path='logs-imdb' --name=imdb-textcnn-static-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=False
