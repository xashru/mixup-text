# Random
python train_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-none

python train_mixup_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-embed --method=embed

python train_mixup_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-sent --method=sent

python train_mixup_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-dense --method=dense

# nonstatic
python train_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-nonstatic-none --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-nonstatic-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-nonstatic-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-nonstatic-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=True

# static
python train_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-static-none --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-static-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-static-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextCNN --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs-dbpedia' --name=dbpedia-textcnn-static-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=False
