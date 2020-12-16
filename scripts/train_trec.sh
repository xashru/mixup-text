# Random
python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-none

python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-embed --method=embed

python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-sent --method=sent

python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-dense --method=dense

# nonstatic
python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-nonstatic-none --w2v-file=data/glove.pickle --fine-tune=True

python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-nonstatic-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=True

python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-nonstatic-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=True

python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-nonstatic-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=True

# static
python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-static-none --w2v-file=data/glove.pickle --fine-tune=False

python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-static-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=False

python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-static-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=False

python train.py --model=TextCNN --task=trec --save-path='logs-trec' --name=trec-textcnn-static-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=False
