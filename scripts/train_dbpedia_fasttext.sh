# Random
python train_multiple.py --model=FastText --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs/logs-dbpedia' --name=dbpedia-FastText-none

python train_mixup_multiple.py --model=FastText --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs/logs-dbpedia' --name=dbpedia-FastText-embed --method=embed


# nonstatic
python train_multiple.py --model=FastText --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs/logs-dbpedia' --name=dbpedia-FastText-nonstatic-none --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=FastText --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs/logs-dbpedia' --name=dbpedia-FastText-nonstatic-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=True


# static
python train_multiple.py --model=FastText --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs/logs-dbpedia' --name=dbpedia-FastText-static-none --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=FastText --task=dbpedia --save-path='../drive/My Drive/research/mixup-text/logs/logs-dbpedia' --name=dbpedia-FastText-static-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=False
