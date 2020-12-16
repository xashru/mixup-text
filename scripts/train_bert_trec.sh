# nonstatic
python train_multiple.py --model=bert-base-uncased --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-bert-base-uncased-nonstatic-none --fine-tune=True --dropout=0.5

python train_multiple.py --model=bert-base-uncased --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-bert-base-uncased-nonstatic-embed --method=embed --fine-tune=True --dropout=0

python train_multiple.py --model=bert-base-uncased --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-bert-base-uncased-nonstatic-sent --method=sent --fine-tune=True --dropout=0

python train_multiple.py --model=bert-base-uncased --task=trec --save-path='../drive/My Drive/research/mixup-text/logs-trec' --name=trec-bert-base-uncased-nonstatic-dense --method=dense --fine-tune=True --dropout=0
